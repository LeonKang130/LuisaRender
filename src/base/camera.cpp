//
// Created by Mike on 2021/12/8.
//

#include <random>

#include <base/scene.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/transform.h>
#include <sdl/scene_node_desc.h>
#include <base/camera.h>

namespace luisa::render {

[[nodiscard]] static auto default_filter_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__camera_default_filter__", SceneNodeTag::FILTER};
        d.define(SceneNodeTag::FILTER, "Box", {});
        return &d;
    }();
    return desc;
}

[[nodiscard]] static auto default_transform_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__camera_default_transform__", SceneNodeTag::TRANSFORM};
        d.define(SceneNodeTag::TRANSFORM, "Identity", {});
        return &d;
    }();
    return desc;
}

Camera::Camera(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::CAMERA},
      _film{scene->load_film(desc->property_node("film"))},
      _filter{scene->load_filter(desc->property_node_or_default(
          "filter", default_filter_node_desc()))},
      _transform{scene->load_transform(desc->property_node_or_default(
          "transform", default_transform_node_desc()))},
      _shutter_span{desc->property_float2_or_default(
          "shutter_span", lazy_construct([desc] {
              return make_float2(desc->property_float_or_default(
                  "shutter_span", 0.0f));
          }))},
      _shutter_samples{desc->property_uint_or_default("shutter_samples", 0u)},// 0 means default
      _spp{desc->property_uint_or_default("spp", 1024u)} {

    if (_shutter_span.y < _shutter_span.x) [[unlikely]] {
        LUISA_ERROR(
            "Invalid time span: [{}, {}]. [{}]",
            _shutter_span.x, _shutter_span.y,
            desc->source_location().string());
    }
    if (_shutter_span.x != _shutter_span.y) {
        if (_shutter_samples == 0u) {
            _shutter_samples = std::max(_spp, 1024u);
        } else if (_shutter_samples > _spp) {
            LUISA_WARNING(
                "Too many shutter samples ({}), "
                "clamping to samples per pixel ({}). [{}]",
                _shutter_samples, _spp,
                desc->source_location().string());
            _shutter_samples = _spp;
        }
        auto shutter_time_points = desc->property_float_list_or_default("shutter_time_points");
        auto shutter_weights = desc->property_float_list_or_default("shutter_weights");
        if (shutter_time_points.size() != shutter_weights.size()) [[unlikely]] {
            LUISA_ERROR(
                "Number of shutter time points and "
                "number of shutter weights mismatch. [{}]",
                desc->source_location().string());
        }
        if (std::any_of(shutter_weights.cbegin(), shutter_weights.cend(), [](auto w) noexcept {
                return w < 0.0f;
            })) [[unlikely]] {
            LUISA_ERROR(
                "Found negative shutter weight. [{}]",
                desc->source_location().string());
        }
        if (shutter_time_points.empty()) {
            _shutter_points.emplace_back(ShutterPoint{_shutter_span.x, 1.0f});
            _shutter_points.emplace_back(ShutterPoint{_shutter_span.y, 1.0f});
        } else {
            luisa::vector<uint> indices(shutter_time_points.size());
            std::iota(indices.begin(), indices.end(), 0u);
            if (auto iter = std::remove_if(indices.begin(), indices.end(), [&](auto i) noexcept {
                    auto t = shutter_time_points[i];
                    return t < _shutter_span.x || t > _shutter_span.y;
                });
                iter != indices.end()) [[unlikely]] {
                LUISA_WARNING(
                    "Out-of-shutter samples (count = {}) "
                    "are to be removed. [{}]",
                    std::distance(iter, indices.end()),
                    desc->source_location().string());
                indices.erase(iter, indices.end());
            }
            std::sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) noexcept {
                return shutter_time_points[lhs] < shutter_time_points[rhs];
            });
            if (auto iter = std::unique(indices.begin(), indices.end(), [&](auto lhs, auto rhs) noexcept {
                    return shutter_time_points[lhs] == shutter_time_points[rhs];
                });
                iter != indices.end()) [[unlikely]] {
                LUISA_WARNING(
                    "Duplicate shutter samples (count = {}) "
                    "are to be removed. [{}]",
                    std::distance(iter, indices.end()),
                    desc->source_location().string());
                indices.erase(iter, indices.end());
            }
            _shutter_points.reserve(indices.size() + 2u);
            _shutter_points.resize(indices.size());
            std::transform(
                indices.cbegin(), indices.cend(),
                std::back_inserter(_shutter_points), [&](auto i) noexcept {
                    return ShutterPoint{shutter_time_points[i], shutter_weights[i]};
                });
            if (!_shutter_points.empty()) {
                if (auto ts = _shutter_span.x; _shutter_points.front().time > ts) {
                    _shutter_points.insert(
                        _shutter_points.begin(),
                        ShutterPoint{ts, _shutter_points.front().weight});
                }
                if (auto te = _shutter_span.y; _shutter_points.back().time < te) {
                    _shutter_points.emplace_back(
                        ShutterPoint{te, _shutter_points.back().weight});
                }
            }
        }
    }

    // render file
    _file = desc->property_path_or_default(
        "file", std::filesystem::canonical(
                    desc->source_location() ?
                        desc->source_location().file()->parent_path() :
                        std::filesystem::current_path()) /
                    "color.exr");
    if (auto folder = _file.parent_path();
        !std::filesystem::exists(folder)) {
        std::filesystem::create_directories(folder);
    }
}

auto Camera::shutter_weight(float time) const noexcept -> float {
    if (time < _shutter_span.x || time > _shutter_span.y) { return 0.0f; }
    if (_shutter_span.x == _shutter_span.y) { return 1.0f; }
    auto ub = std::upper_bound(
        _shutter_points.cbegin(), _shutter_points.cend(), time,
        [](auto lhs, auto rhs) noexcept { return lhs < rhs.time; });
    auto u = std::distance(_shutter_points.cbegin(), ub);
    auto p0 = _shutter_points[u - 1u];
    auto p1 = _shutter_points[u];
    auto t = (time - p0.time) / (p1.time - p0.time);
    return std::lerp(p0.weight, p1.weight, t);
}

auto Camera::shutter_samples() const noexcept -> vector<ShutterSample> {
    if (_shutter_span.x == _shutter_span.y) {
        ShutterPoint sp{_shutter_span.x, 1.0f};
        return {ShutterSample{sp, _spp}};
    }
    auto duration = _shutter_span.y - _shutter_span.x;
    auto inv_n = 1.0f / static_cast<float>(_shutter_samples);
    std::uniform_real_distribution<float> dist{};
    std::default_random_engine random{std::random_device{}()};
    luisa::vector<ShutterSample> buckets(_shutter_samples);
    for (auto bucket = 0u; bucket < _shutter_samples; bucket++) {
        auto ts = static_cast<float>(bucket) * inv_n * duration;
        auto te = static_cast<float>(bucket + 1u) * inv_n * duration;
        auto a = dist(random);
        auto t = std::lerp(ts, te, a);
        auto w = shutter_weight(t);
        buckets[bucket].point = ShutterPoint{t, w};
    }
    luisa::vector<uint> indices(_shutter_samples);
    std::iota(indices.begin(), indices.end(), 0u);
    std::shuffle(indices.begin(), indices.end(), random);
    auto remainder = _spp % _shutter_samples;
    auto samples_per_bucket = _spp / _shutter_samples;
    for (auto i = 0u; i < remainder; i++) { buckets[indices[i]].spp = samples_per_bucket + 1u; }
    for (auto i = remainder; i < _shutter_samples; i++) { buckets[indices[i]].spp = samples_per_bucket; }
    auto sum_weights = std::accumulate(buckets.cbegin(), buckets.cend(), 0.0, [](auto lhs, auto rhs) noexcept {
        return lhs + rhs.point.weight * rhs.spp;
    });
    if (sum_weights == 0.0) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid shutter samples generated. "
            "Falling back to uniform shutter curve.");
        for (auto &s : buckets) { s.point.weight = 1.0f; }
    } else {
        auto scale = _spp / sum_weights;
        for (auto &s : buckets) {
            s.point.weight = static_cast<float>(s.point.weight * scale);
        }
    }
    return buckets;
}

}// namespace luisa::render