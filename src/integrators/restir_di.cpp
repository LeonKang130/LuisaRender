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
class ReSTIRDirectIllumination final : public ProgressiveIntegrator {

private:
    bool _presample_lights;
    bool _visibility_reuse;
    bool _temporal_reuse;
    bool _spatial_reuse;
    bool _unbiased;
    bool _decorrelate;
    uint _mutation_num;
public:
    ReSTIRDirectIllumination(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _presample_lights{desc->property_bool_or_default("presample_lights", true)},
          _visibility_reuse{desc->property_bool_or_default("visibility_reuse", true)},
          _temporal_reuse{desc->property_bool_or_default("temporal_reuse", true)},
          _spatial_reuse{desc->property_bool_or_default("spatial_reuse", true)},
          _unbiased{desc->property_bool_or_default("unbiased", false)},
          _decorrelate{desc->property_bool_or_default("decorrelate", false)},
          _mutation_num{desc->property_uint_or_default("mutation_num", 1u)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &cb) const noexcept override;
    [[nodiscard]] bool presample_lights() const noexcept { return _presample_lights; }
    [[nodiscard]] bool visibility_reuse() const noexcept { return _visibility_reuse; }
    [[nodiscard]] bool temporal_reuse() const noexcept { return _temporal_reuse; }
    [[nodiscard]] bool spatial_reuse() const noexcept { return _spatial_reuse; }
    [[nodiscard]] bool unbiased() const noexcept { return _unbiased; }
    [[nodiscard]] bool decorrelate() const noexcept { return _decorrelate; }
    [[nodiscard]] uint mutation_num() const noexcept { return _mutation_num; }
};

class ReSTIRDirectIlluminationInstance final : public ProgressiveIntegrator::Instance {

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

    class LightSubsetBuffer {
    private:
        Buffer<float3> _u_sample;
    public:
        static auto constexpr subset_size = 64u;
        static auto constexpr subset_num = 128u;
        explicit LightSubsetBuffer(const Spectrum::Instance *spectrum) noexcept {
            auto &&device = spectrum->pipeline().device();
            _u_sample = device.create_buffer<float3>(subset_num * subset_size);
        }
        void write(Expr<uint> index, Expr<float> u_sel, Expr<float2> u_light) const noexcept {
            _u_sample->write(index, make_float3(u_sel, u_light));
        }
        [[nodiscard]] auto read(Expr<uint> index) const noexcept {
            auto u_sample = _u_sample->read(index);
            return std::make_pair(u_sample.x, u_sample.yz());
        }
        [[nodiscard]] auto sample(Expr<float> u1, Expr<float> u2) const noexcept {
            auto subset_index = clamp(u1 * static_cast<float>(subset_num), 0.f, subset_num - 1.f).cast<uint>();
            auto subset_offset = clamp(u2 * static_cast<float>(subset_size), 0.f, subset_size - 1.f).cast<uint>();
            return read(subset_index * subset_size + subset_offset);
        }
    };

    class ReservoirBuffer {
    public:
        struct Reservoir {
            Float3 u_sample;
            Float m;
            Float p_hat;
            Float w;
            UInt age; // samples older than 15 (255 in RTX DI) frames are discarded
            [[nodiscard]] static auto zero() noexcept { return Reservoir{make_float3(0.f), 0.f, 0.f, 0.f, 0u}; }
            [[nodiscard]] auto update(Reservoir const &r, Expr<float> u) const noexcept {
                auto rr = r;
                auto w1 = r.w * r.m, w2 = w * m;
                auto w_sum = w1 + w2;
                $if (u * w_sum > w1) {
                    rr.u_sample = u_sample;
                    rr.p_hat = p_hat;
                    rr.age = age;
                };
                rr.m = r.m + m;
                rr.w = ite(rr.m > 0.f, w_sum / rr.m, 0.f);
                return rr;
            }
        };
    private:
        Buffer<float3> _u_sample;
        Buffer<float> _m;
        Buffer<float> _p_hat;
        Buffer<float> _w;
        Buffer<uint> _age;
    public:
        ReservoirBuffer(const Spectrum::Instance *spectrum, size_t size) noexcept {
            auto &&device = spectrum->pipeline().device();
            _u_sample = device.create_buffer<float3>(size);
            _m = device.create_buffer<float>(size);
            _p_hat = device.create_buffer<float>(size);
            _w = device.create_buffer<float>(size);
            _age = device.create_buffer<uint>(size);
        }
        [[nodiscard]] auto read(Expr<uint> index) const noexcept {
            auto r = Reservoir::zero();
            r.u_sample = _u_sample->read(index);
            r.m = _m->read(index);
            r.p_hat = _p_hat->read(index);
            r.w = _w->read(index);
            r.age = _age->read(index);
            return r;
        }
        void write(Expr<uint> index, Reservoir const &r) const noexcept {
            _u_sample->write(index, r.u_sample);
            _m->write(index, r.m);
            _p_hat->write(index, r.p_hat);
            _w->write(index, r.w);
            _age->write(index, r.age);
        }
    };


private:
    luisa::unique_ptr<VisibilityBuffer> _visibility_buffer;
    luisa::unique_ptr<LightSubsetBuffer> _light_subset_buffer;
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
        // initialize ReSTIRDirectIllumination buffers
        if (!_visibility_buffer) {
            _visibility_buffer = luisa::make_unique<VisibilityBuffer>(pipeline().spectrum(), pixel_count);
        }
        if (!_light_subset_buffer) {
            _light_subset_buffer = luisa::make_unique<LightSubsetBuffer>(pipeline().spectrum());
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
        auto presample_light_shader = compile_async<1>(device, [&](UInt frame_index) noexcept {
            set_block_size(256u);
            auto i = dispatch_id().x;
            sampler()->start(make_uint2(i, i), frame_index);
            _light_subset_buffer->write(i, sampler()->generate_1d(), sampler()->generate_2d());
        });
        auto sample_light_shader = compile_async<2>(device, [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _sample_light(camera, frame_index, pixel_id, time);
        });
        auto temporal_reuse_shader = compile_async<2>(device, [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _temporal_reuse(camera, frame_index, pixel_id, time);
        });
        auto swap_buffer_shader = compile_async<2>(device, [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto index = pixel_id.x + pixel_id.y * resolution.x;
            auto r = _reservoir_buffer_out->read(index);
            _reservoir_buffer->write(index, r);
        });
        auto decorrelate_shader = compile_async<2>(device, [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _decorrelate_samples(camera, frame_index, pixel_id, time);
        });
        auto spatial_reuse_shader = compile_async<2>(device, [&](UInt frame_index, UInt pass_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            if (node<ReSTIRDirectIllumination>()->unbiased()) {
                _unbiased_spatial_reuse(camera, frame_index, pass_index, pixel_id, time);
            } else {
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
        presample_light_shader.get().set_name("presample_light");
        sample_light_shader.get().set_name("sample_light");
        temporal_reuse_shader.get().set_name("temporal_reuse");
        swap_buffer_shader.get().set_name("swap_buffer");
        decorrelate_shader.get().set_name("decorrelate");
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
                if (node<ReSTIRDirectIllumination>()->presample_lights()) {
                    command_buffer << presample_light_shader.get()(sample_id).dispatch(LightSubsetBuffer::subset_num * LightSubsetBuffer::subset_size);
                }
                command_buffer << sample_light_shader.get()(sample_id, s.point.time).dispatch(resolution);
                if (node<ReSTIRDirectIllumination>()->temporal_reuse() && sample_id != 0u) {
                    command_buffer << temporal_reuse_shader.get()(sample_id, s.point.time).dispatch(resolution);
                    if (node<ReSTIRDirectIllumination>()->decorrelate()) {
                        command_buffer << decorrelate_shader.get()(sample_id, s.point.time).dispatch(resolution);
                    }
                }
                command_buffer << swap_buffer_shader.get()().dispatch(resolution);
                if (node<ReSTIRDirectIllumination>()->spatial_reuse()) {
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
        auto u_subset = def(0.f);
        if (node<ReSTIRDirectIllumination>()->presample_lights()) {
            sampler()->start($block_id.xy(), frame_index);
            u_subset = sampler()->generate_1d();
        }
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
                    auto u_sel = def(0.f);
                    auto u_light = def(make_float2(0.f));
                    if (node<ReSTIRDirectIllumination>()->presample_lights()) {
                        auto [uu_sel, uu_light] = _light_subset_buffer->sample(u_subset, sampler()->generate_1d());
                        u_sel = uu_sel;
                        u_light = uu_light;
                    } else {
                        u_sel = sampler()->generate_1d();
                        u_light = sampler()->generate_2d();
                    }
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    $outline {
                        light_sample = light_sampler()->sample(
                            *it, u_sel, u_light, swl, time);
                    };
                    // direct lighting
                    auto candidate = ReservoirBuffer::Reservoir::zero();
                    candidate.u_sample = make_float3(u_sel, u_light);
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
            if (node<ReSTIRDirectIllumination>()->visibility_reuse()) {
                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                auto occluded = def(true);
                $outline {
                    light_sample = light_sampler()->sample(
                        *it, r.u_sample.x, r.u_sample.yz(), swl, time);
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
    void _temporal_reuse(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
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
            $if(candidate.age > 15u) {
                $break;
            };
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                light_sample = light_sampler()->sample(
                    *it, candidate.u_sample.x, candidate.u_sample.yz(), swl, time);
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
        r.age += 1u;
        _reservoir_buffer_out->write(index, r);
    }
    void _decorrelate_samples(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto resolution = camera->film()->node()->resolution();
        auto index = pixel_id.x + pixel_id.y * resolution.x;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->read_ray(index);
        auto hit = _visibility_buffer->read_hit(index);
        auto r = _reservoir_buffer_out->read(index);
        $loop {
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto wo = -ray->direction();
            // miss
            $if (!it->valid() | !it->shape().has_surface()) {
                $break;
            };
            $if (r.p_hat == 0.f | r.w == 0.f) {
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
                // perform MCMC mutations
                auto constexpr s1 = 1.f / 1024.f, s2 = 1.f / 64.f;
                auto mutate_1d = [&](auto u) noexcept {
                    auto uu = abs(2.f * u - 1.f);
                    auto s = s1 * exp(-log(s1 / s2) * uu);
                    return ite(u < .5f, 1.f, -1.f) * s;
                };
                auto mutate_2d = [&](auto u) noexcept {
                    auto radius = s2 * sqrt(-2.f * log(max(std::numeric_limits<float>::min(), 1.f - u.x)));
                    auto phi = 2.f * pi * u.y;
                    return radius * make_float2(cos(phi), sin(phi));
                };
                $for (i, 0u, node<ReSTIRDirectIllumination>()->mutation_num()) {
                    auto uu_sel = clamp(r.u_sample.x + mutate_1d(sampler()->generate_1d()), 0.f, 1.f);
                    auto uu_light = clamp(r.u_sample.yz() + mutate_2d(sampler()->generate_2d()), 0.f, 1.f);
                    auto light_sample = LightSampler::Sample::zero(swl.dimension());
                    auto occluded = def(false);
                    $outline {
                        light_sample = light_sampler()->sample(
                            *it, uu_sel, uu_light, swl, time);
                    };
                    if (node<ReSTIRDirectIllumination>()->visibility_reuse()) {
                        occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                    }
                    $if (light_sample.eval.pdf > 0.f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if (eval.pdf > 0.f) {
                            auto Li = eval.f * light_sample.eval.L;
                            auto p_hat = spectrum->cie_y(swl, Li);
                            $if (sampler()->generate_1d() < min(1.f, p_hat / r.p_hat)) {
                                r.u_sample = make_float3(uu_sel, uu_light);
                                r.p_hat = p_hat;
                            };
                        };
                    };
                };
            });
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
                            *it, candidate.u_sample.x, candidate.u_sample.yz(), swl, time);
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
                            *it, candidate.u_sample.x, candidate.u_sample.yz(), swl, time);
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
            if (node<ReSTIRDirectIllumination>()->visibility_reuse()) {
                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                $outline {
                    light_sample = light_sampler()->sample(
                        *it, r.u_sample.x, r.u_sample.yz(), swl, time);
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
                            *neighbor_it, r.u_sample.x, r.u_sample.yz(), swl, time);
                    };
                    $if (light_sample.eval.pdf > 0.f) {
                        auto occluded = def(false);
                        if (node<ReSTIRDirectIllumination>()->visibility_reuse()) {
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
                    *it, r.u_sample.x, r.u_sample.yz(), swl, time);
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
                if (node<ReSTIRDirectIllumination>()->visibility_reuse()) {
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

luisa::unique_ptr<Integrator::Instance> ReSTIRDirectIllumination::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<ReSTIRDirectIlluminationInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ReSTIRDirectIllumination)