//
// Created by Leon Kang on 2024/7/25.
//

#include <dsl/syntax.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <util/thread_pool.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

template<uint dim, typename F>
[[nodiscard]] auto compile_async(Device &device, F &&f) noexcept {
    using namespace compute;
    auto kernel = [&] {
        if constexpr (dim == 1u) {
            return Kernel1D{f};
        } else if constexpr (dim == 2u) {
            return Kernel2D{f};
        } else if constexpr (dim == 3u) {
            return Kernel3D{f};
        } else {
            static_assert(always_false_v<F>, "Invalid dimension.");
        }
    }();
    ShaderOption o{};
    o.enable_debug_info = true;
    return global_thread_pool().async([&device, o, kernel] {
        return device.compile(kernel, o);
    });
}

/*
 * Accompanied light sampling technique must be independent of shading points
 * so that light samples can be stored as three float point numbers
*/
class ReSTIR final : public ProgressiveIntegrator {

private:
    bool _visibility_reuse;
    bool _temporal_reuse;
    bool _spatial_reuse;
    bool _unbiased;
public:
    ReSTIR(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _visibility_reuse{desc->property_bool_or_default("visibility_reuse", true)},
          _temporal_reuse{desc->property_bool_or_default("temporal_reuse", true)},
          _spatial_reuse{desc->property_bool_or_default("spatial_reuse", true)},
          _unbiased{desc->property_bool_or_default("unbiased", false)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &cb) const noexcept override;
    [[nodiscard]] bool visibility_reuse() const noexcept { return _visibility_reuse; }
    [[nodiscard]] bool temporal_reuse() const noexcept { return _temporal_reuse; }
    [[nodiscard]] bool spatial_reuse() const noexcept { return _spatial_reuse; }
    [[nodiscard]] bool unbiased() const noexcept { return _unbiased; }
};

class ReSTIRInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

    class VisibilityBuffer {
    private:
        Buffer<Ray> _ray;
        Buffer<Hit> _hit;
        Buffer<float> _w;
    public:
        VisibilityBuffer(const Spectrum::Instance *spectrum, size_t size) noexcept {
            auto &&device = spectrum->pipeline().device();
            _ray = device.create_buffer<Ray>(size);
            _hit = device.create_buffer<Hit>(size);
            _w = device.create_buffer<float>(size);
        }
        [[nodiscard]] auto read_ray(Expr<uint> index) const noexcept { return _ray->read(index); }
        [[nodiscard]] auto read_hit(Expr<uint> index) const noexcept { return _hit->read(index); }
        [[nodiscard]] auto read_camera_weight(Expr<uint> index) const noexcept { return _w->read(index); }
        void write_ray(Expr<uint> index, Expr<Ray> ray) noexcept { _ray->write(index, ray); }
        void write_hit(Expr<uint> index, Expr<Hit> hit) noexcept { _hit->write(index, hit); }
        void write_camera_weight(Expr<uint> index, Expr<float> w) noexcept { _w->write(index, w); }
    };

    class ReservoirBuffer {
    public:
        struct Reservoir {
            Float u_sel;
            Float2 u_light;
            Float m;
            Float p_hat;
            Float w;
            [[nodiscard]] static auto zero() noexcept { return Reservoir{0.f, make_float2(0.f), 0.f, 0.f, 0.f}; }
            [[nodiscard]] auto update(Reservoir const &r, Expr<float> u) const noexcept {
                auto rr = Reservoir::zero();
                rr.m = r.m + m;
                auto w1 = r.w * r.m, w2 = w * m;
                auto w_sum = w1 + w2;
                $if (u * w_sum < w1) {
                    rr.u_sel = r.u_sel;
                    rr.u_light = r.u_light;
                    rr.p_hat = r.p_hat;
                }
                $else {
                    rr.u_sel = u_sel;
                    rr.u_light = u_light;
                    rr.p_hat = p_hat;
                };
                rr.w = ite(rr.m > 0.f, w_sum / rr.m, 0.f);
                return rr;
            }
        };
    private:
        Buffer<float> _u_sel;
        Buffer<float2> _u_light;
        Buffer<float> _m;
        Buffer<float> _p_hat;
        Buffer<float> _w;
    public:
        ReservoirBuffer(const Spectrum::Instance *spectrum, size_t size) noexcept {
            auto &&device = spectrum->pipeline().device();
            _u_sel = device.create_buffer<float>(size);
            _u_light = device.create_buffer<float2>(size);
            _m = device.create_buffer<float>(size);
            _p_hat = device.create_buffer<float>(size);
            _w = device.create_buffer<float>(size);
        }
        [[nodiscard]] auto read(Expr<uint> index) const noexcept {
            auto r = Reservoir::zero();
            r.u_sel = _u_sel->read(index);
            r.u_light = _u_light->read(index);
            r.m = _m->read(index);
            r.p_hat = _p_hat->read(index);
            r.w = _w->read(index);
            return r;
        }
        void write(Expr<uint> index, Reservoir const &r) const noexcept {
            _u_sel->write(index, r.u_sel);
            _u_light->write(index, r.u_light);
            _m->write(index, r.m);
            _p_hat->write(index, r.p_hat);
            _w->write(index, r.w);
        }
    };


private:
    luisa::unique_ptr<VisibilityBuffer> _visibility_buffer;
    luisa::unique_ptr<ReservoirBuffer> _reservoir_buffer;
    luisa::unique_ptr<ReservoirBuffer> _reservoir_buffer_out;
protected:
    void _render_one_camera(CommandBuffer &command_buffer,
                            Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        // initialize ReSTIR buffers
        if (!_visibility_buffer) {
            _visibility_buffer = luisa::make_unique<VisibilityBuffer>(pipeline().spectrum(), pixel_count);
        }
        if (!_reservoir_buffer) {
            _reservoir_buffer = luisa::make_unique<ReservoirBuffer>(pipeline().spectrum(), pixel_count);
        }
        if (!_reservoir_buffer_out) {
            _reservoir_buffer_out = luisa::make_unique<ReservoirBuffer>(pipeline().spectrum(), pixel_count);
        }
        command_buffer << compute::synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        auto &&device = pipeline().device();
        auto ray_buffer = device.create_buffer<Ray>(pixel_count);
        auto hit_buffer = device.create_buffer<Hit>(pixel_count);
        Clock clock_compile;
        auto sample_light_shader = compile_async<2>(device, [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _sample_light(camera, frame_index, pixel_id, time);
        });
        auto temporal_reuse_shader = compile_async<2>(device, [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            temporal_reuse(camera, frame_index, pixel_id, time);
        });
        auto swap_buffer_shader = compile_async<2>(device, [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto index = pixel_id.x + pixel_id.y * resolution.x;
            auto r = _reservoir_buffer_out->read(index);
            _reservoir_buffer->write(index, r);
        });
        auto spatial_reuse_shader = compile_async<2>(device, [&](UInt frame_index, UInt pass_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            if(node<ReSTIR>()->unbiased()) {
                _unbiased_spatial_reuse(camera, frame_index, pass_index, pixel_id, time);
            }
            else {
                _spatial_reuse(camera, frame_index, pass_index, pixel_id, time);
            }
        });
        auto render_shader = compile_async<2>(device, [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        });
        // wait for the compilation of all shaders
        sample_light_shader.get().set_name("sample_light");
        temporal_reuse_shader.get().set_name("temporal_reuse");
        swap_buffer_shader.get().set_name("swap_buffer");
        spatial_reuse_shader.get().set_name("spatial_reuse");
        render_shader.get().set_name("render");
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            for (auto i = 0u; i < s.spp; i++) {
                // camera->film()->clear(command_buffer);
                command_buffer << sample_light_shader.get()(sample_id, s.point.time).dispatch(resolution);
                if (node<ReSTIR>()->temporal_reuse() && sample_id != 0u) {
                    command_buffer << temporal_reuse_shader.get()(sample_id, s.point.time).dispatch(resolution);
                }
                command_buffer << swap_buffer_shader.get()().dispatch(resolution);
                if (node<ReSTIR>()->spatial_reuse()) {
                    for (auto j = 0u; j < 2u; j++) {
                        command_buffer << spatial_reuse_shader.get()(sample_id, j, s.point.time).dispatch(resolution);
                        command_buffer << swap_buffer_shader.get()().dispatch(resolution);
                    }
                }
                command_buffer << render_shader.get()(sample_id++, s.point.time, s.point.weight)
                                      .dispatch(resolution);
                command_buffer << swap_buffer_shader.get()().dispatch(resolution);
                dispatch_count++;
                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
                auto dispatches_per_commit = 4u;
                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }
    void _sample_light(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto cs = camera->generate_ray(pixel_id, time, make_float2(.5f), make_float2(.5f));
        auto ray = cs.ray;
        auto hit = pipeline().geometry()->trace_closest(cs.ray);
        _visibility_buffer->write_ray(index, ray);
        _visibility_buffer->write_hit(index, hit);
        _visibility_buffer->write_camera_weight(index, cs.weight);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto r = ReservoirBuffer::Reservoir::zero();
        $loop {
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // some preparations
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // sample light candidates
                $for (x, 0u, 32u) {
                    auto u_sel = sampler()->generate_1d();
                    auto u_light = sampler()->generate_2d();
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    $outline {
                        light_sample = light_sampler()->sample(
                            *it, u_sel, u_light, swl, time);
                    };
                    // direct lighting
                    auto candidate = ReservoirBuffer::Reservoir::zero();
                    candidate.u_sel = u_sel;
                    candidate.u_light = u_light;
                    candidate.m = 1.f;
                    $if (light_sample.eval.pdf > 0.0f) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if (eval.pdf > 0.f) {
                            auto Li = eval.f * light_sample.eval.L;
                            // update reservoir
                            candidate.p_hat = spectrum->cie_y(swl, Li);
                            candidate.w = candidate.p_hat / light_sample.eval.pdf;
                        };
                    };
                    r = r.update(candidate, sampler()->generate_1d());
                };
            });
            // visibility test
            if (node<ReSTIR>()->visibility_reuse()) {
                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                auto occluded = def(true);
                $outline {
                    light_sample = light_sampler()->sample(
                        *it, r.u_sel, r.u_light, swl, time);
                };
                $if (light_sample.eval.pdf > 0.f &
                     light_sample.eval.L.any([](auto x) { return x > 0.f; })) {
                    occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                };
                $if (occluded) {
                    r.p_hat = 0.f;
                };
            }
            $break;
        };
        _reservoir_buffer_out->write(index, r);
    }
    void temporal_reuse(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->read_ray(index);
        auto hit = _visibility_buffer->read_hit(index);
        auto r = _reservoir_buffer_out->read(index);
        auto candidate = _reservoir_buffer->read(index);
        auto m_capped = min(candidate.m, 20.f * r.m);
        candidate.m = m_capped;
        $loop {
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                light_sample = light_sampler()->sample(
                    *it, candidate.u_sel, candidate.u_light, swl, time);
            };
            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // some preparations
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // re-evaluate target function for temporal neighbor's sample
                $if (light_sample.eval.pdf > 0.f & candidate.p_hat > 0.f) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    $if (eval.pdf > 0.f) {
                        auto Li = eval.f * light_sample.eval.L;
                        auto w = spectrum->cie_y(swl, Li) / candidate.p_hat;
                        candidate.p_hat *= w;
                        candidate.w *= w;
                    }
                    $else {
                        candidate.p_hat = candidate.w = 0.f;
                    };
                }
                $else {
                    candidate.p_hat = candidate.w = 0.f;
                };
            });
            r = r.update(candidate, sampler()->generate_1d());
            $break;
        };
        _reservoir_buffer_out->write(index, r);
    }
    void _spatial_reuse(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint> pass_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index << 1u | pass_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->read_ray(index);
        auto hit = _visibility_buffer->read_hit(index);
        auto r = _reservoir_buffer->read(index);
        $loop {
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            auto depth = distance(ray->origin(), it->p());
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            auto surface_tag = it->shape().surface_tag();
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // some preparations
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // re-evaluate target function of spatial neighbors' samples
                $for (i, 0u, 5u) {
                    auto neighbor_offset = 20.f * sample_uniform_disk_concentric(sampler()->generate_2d());
                    auto neighbor_pixel_id = make_uint2(clamp(make_float2(pixel_id) + neighbor_offset, 0.f, make_float2(resolution - 1u)));
                    auto neighbor_ray = _visibility_buffer->read_ray(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    auto neighbor_hit = _visibility_buffer->read_hit(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    auto neighbor_it = pipeline().geometry()->interaction(neighbor_ray, neighbor_hit);
                    auto neighbor_depth = distance(ray->origin(), neighbor_it->p());
                    $if (!neighbor_it->valid() | !neighbor_it->shape().has_surface()) {
                        $continue;
                    };
                    $if (dot(it->shading().n(), neighbor_it->shading().n()) < 0.9f | abs(depth - neighbor_depth) / depth > 0.1f) {
                        $continue;
                    };
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    auto candidate = _reservoir_buffer->read(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    $outline {
                        light_sample = light_sampler()->sample(
                            *it, candidate.u_sel, candidate.u_light, swl, time);
                    };
                    $if (light_sample.eval.pdf > 0.f & candidate.p_hat > 0.f) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if (eval.pdf > 0.f) {
                            auto Li = eval.f * light_sample.eval.L;
                            auto w = spectrum->cie_y(swl, Li) / candidate.p_hat;
                            candidate.p_hat *= w;
                            candidate.w *= w;
                        }
                        $else {
                            candidate.p_hat = candidate.w = 0.f;
                        };
                    }
                    $else {
                        candidate.p_hat = candidate.w = 0.f;
                    };
                    r = r.update(candidate, sampler()->generate_1d());
                };
            });
            $break;
        };
        _reservoir_buffer_out->write(index, r);
    }
    void _unbiased_spatial_reuse(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint> pass_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index << 1u | pass_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->read_ray(index);
        auto hit = _visibility_buffer->read_hit(index);
        auto r = _reservoir_buffer->read(index);
        $loop {
            ArrayVar<Ray, 3u> neighbor_rays;
            ArrayVar<Hit, 3u> neighbor_hits;
            ArrayVar<Float, 3u> neighbor_ms;
            auto neighbor_count = def(0u);
            auto z = r.m;
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            auto depth = distance(ray->origin(), it->p());
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            auto surface_tag = it->shape().surface_tag();
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // some preparations
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // re-evaluate target function of spatial neighbors' samples
                $for (i, 0u, 3u) {
                    auto neighbor_offset = 20.f * sample_uniform_disk_concentric(sampler()->generate_2d());
                    auto neighbor_pixel_id = make_uint2(clamp(make_float2(pixel_id) + neighbor_offset, 0.f, make_float2(resolution - 1u)));
                    auto neighbor_ray = _visibility_buffer->read_ray(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    auto neighbor_hit = _visibility_buffer->read_hit(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    auto neighbor_it = pipeline().geometry()->interaction(neighbor_ray, neighbor_hit);
                    auto neighbor_depth = distance(ray->origin(), neighbor_it->p());
                    $if (!neighbor_it->valid() | !neighbor_it->shape().has_surface()) {
                        $continue;
                    };
                    $if (dot(it->shading().n(), neighbor_it->shading().n()) < 0.9f | abs(depth - neighbor_depth) / depth > 0.1f) {
                        $continue;
                    };
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    auto candidate = _reservoir_buffer->read(neighbor_pixel_id.x + neighbor_pixel_id.y * resolution.x);
                    $outline {
                        light_sample = light_sampler()->sample(
                            *it, candidate.u_sel, candidate.u_light, swl, time);
                    };
                    $if (light_sample.eval.pdf > 0.f & candidate.p_hat > 0.f) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if (eval.pdf > 0.f) {
                            auto Li = eval.f * light_sample.eval.L;
                            auto w = spectrum->cie_y(swl, Li) / candidate.p_hat;
                            candidate.p_hat *= w;
                            candidate.w *= w;
                        }
                        $else {
                            candidate.p_hat = candidate.w = 0.f;
                        };
                    }
                    $else {
                        candidate.p_hat = candidate.w = 0.f;
                    };
                    r = r.update(candidate, sampler()->generate_1d());
                    neighbor_rays[neighbor_count] = neighbor_ray;
                    neighbor_hits[neighbor_count] = neighbor_hit;
                    neighbor_ms[neighbor_count] = candidate.m;
                    neighbor_count += 1u;
                };
            });
            if (node<ReSTIR>()->visibility_reuse()) {
                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                $outline {
                    light_sample = light_sampler()->sample(
                        *it, r.u_sel, r.u_light, swl, time);
                };
                auto occluded = def(true);
                $if (light_sample.eval.pdf > 0.f) {
                    occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                };
                $if (occluded) {
                    r.p_hat = r.w = 0.f;
                };
            }
            $for (i, 0u, neighbor_count) {
                auto neighbor_ray = neighbor_rays[i];
                auto neighbor_hit = neighbor_hits[i];
                auto neighbor_it = pipeline().geometry()->interaction(neighbor_ray, neighbor_hit);
                auto wo = -neighbor_ray->direction();
                auto surface_tag = neighbor_it->shape().surface_tag();
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *neighbor_it, swl, wo, 1.f, time);
                });
                call.execute([&](auto closure) {
                    // some preparations
                    if (auto dispersive = closure->is_dispersive()) {
                        $if (*dispersive) { swl.terminate_secondary(); };
                    }
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    $outline {
                        light_sample = light_sampler()->sample(
                            *neighbor_it, r.u_sel, r.u_light, swl, time);
                    };
                    $if (light_sample.eval.pdf > 0.f) {
                        auto occluded = def(false);
                        if (node<ReSTIR>()->visibility_reuse()) {
                            occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                        }
                        $if (!occluded) {
                            auto wi = light_sample.shadow_ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            $if (eval.pdf > 0.f) {
                                auto Li = eval.f * light_sample.eval.L;
                                $if (Li.any([](auto x) { return x > 0.f; })) {
                                    z += neighbor_ms[i];
                                };
                            };
                        };
                    };
                });
            };
            r.w *= r.m / z;
            $break;
        };
        _reservoir_buffer_out->write(index, r);
    }
    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};
        auto ray = _visibility_buffer->read_ray(index);
        auto hit = _visibility_buffer->read_hit(index);
        $loop {
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            // miss
            $if (!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += eval.L;
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if (it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += eval.L;
                };
            }

            // compute direct lighting
            $if (!it->shape().has_surface()) { $break; };
            auto r = _reservoir_buffer->read(index);
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                light_sample = light_sampler()->sample(
                    *it, r.u_sel, r.u_light, swl, time);
            };
            // trace shadow ray
            $if (light_sample.eval.pdf > 0.f &
                 light_sample.eval.L.any([](auto x) { return x > 0.f; })) {
                $if (pipeline().geometry()->intersect_any(light_sample.shadow_ray)) {
                    r.p_hat = r.w = 0.f;
                };
            }
            $else {
                r.p_hat = r.w = 0.f;
            };
            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto alpha_skip = def(false);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // some preparations
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // direct lighting
                $if (r.w > 0.f) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    $if (eval.pdf > 0.f) {
                        Li += eval.f * light_sample.eval.L * r.w / r.p_hat;
                    };
                };
                if (node<ReSTIR>()->visibility_reuse()) {
                    _reservoir_buffer_out->write(index, r);
                }
            });
            $if (!alpha_skip) {
                $break;
            };
        };
        auto w = _visibility_buffer->read_camera_weight(index);
        return spectrum->srgb(swl, Li * w);
    }
};

luisa::unique_ptr<Integrator::Instance> ReSTIR::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<ReSTIRInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ReSTIR)