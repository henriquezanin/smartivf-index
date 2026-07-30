#include "smartivf/smartivf.hpp"

#include "smartivf/build_partition.hpp"
#include "smartivf/compute_quantile.hpp"
#include "smartivf/quantize.hpp"
#include "smartivf/range_decompose.hpp"
#include "smartivf/sfc/z_order.hpp"
#include "smartivf/utils/fvecs.hpp"
#include "smartivf/utils/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

#include <faiss/utils/distances.h>
#include <omp.h>

namespace smartivf {

namespace {

[[nodiscard]] std::unique_ptr<sfc::SFC>
make_sfc(SFCType type, std::size_t dims, std::uint8_t bits) {
    switch (type) {
        case SFCType::ZOrder: return std::make_unique<sfc::ZOrder>(dims, bits);
    }
    return nullptr;
}

// FAISS-accelerated inner-product / L2.
[[nodiscard]] float ip(std::span<const float> a, std::span<const float> b) noexcept {
    return faiss::fvec_inner_product(a.data(), b.data(), a.size());
}
[[nodiscard]] float norm_sq(std::span<const float> v) noexcept {
    return faiss::fvec_norm_L2sqr(v.data(), v.size());
}

// L2 squared distance via the identity ‖q-x‖² = ‖q‖² + ‖x‖² − 2 q·x, with both
// norms already available: q_norm_sq is computed once per query, x_norm_sq at
// build time.
[[nodiscard]] float l2_with_norms(std::span<const float> q,
                                  std::span<const float> x,
                                  float q_norm_sq,
                                  float x_norm_sq) noexcept {
    return q_norm_sq + x_norm_sq - 2.0f * ip(q, x);
}

void l2_normalise(std::span<float> v) noexcept {
    const float n2 = norm_sq(v);
    if (n2 > 0.0f) {
        const float inv = 1.0f / std::sqrt(n2);
        for (auto& x : v) x *= inv;
    }
}

// Locate the SFC-value slice of `list` that falls inside any of the given
// intervals. `list` is sorted ascending by sfc_value, so one lower_bound plus
// one upper_bound per interval suffices (O(|intervals| · log |list|)).
void collect_by_binary_search(std::span<const std::shared_ptr<Object>> list,
                              std::span<const SFCInterval> intervals,
                              std::vector<std::shared_ptr<Object>>& out) {
    for (const auto& iv : intervals) {
        const auto lo = std::lower_bound(list.begin(), list.end(), iv.start,
            [](const std::shared_ptr<Object>& o, std::uint64_t v) { return o->sfc_value < v; });
        const auto hi = std::upper_bound(list.begin(), list.end(), iv.end,
            [](std::uint64_t v, const std::shared_ptr<Object>& o) { return v < o->sfc_value; });
        out.insert(out.end(), lo, hi);
    }
}

[[nodiscard]] bool scalar_matches(std::span<const float> attrs,
                                  std::span<const std::array<float, 2>> ranges) noexcept {
    const std::size_t dims = std::min(attrs.size(), ranges.size());
    for (std::size_t d = 0; d < dims; ++d) {
        if (attrs[d] < ranges[d][0] || attrs[d] > ranges[d][1]) return false;
    }
    return true;
}

}  // namespace

SmartIVF::SmartIVF(const SmartIVFConfig& cfg)
    : cfg_(cfg),
      sfc_(make_sfc(cfg.sfc, static_cast<std::size_t>(cfg.scalar_dimensions),
                    cfg.scalar_bits_per_dimension)) {
    if (!sfc_) {
        throw std::invalid_argument(std::format(
            "SmartIVF: unsupported SFC type {}", to_string(cfg.sfc)));
    }
}

Result<Index>
SmartIVF::build_index(const std::string& fvecs_file,
                     const std::string& scalars_file) const {
    auto embeddings = utils::read_fvecs(fvecs_file);
    if (!embeddings) return std::unexpected(embeddings.error());
    auto scalars = utils::read_scalar_csv(scalars_file, false);
    if (!scalars) return std::unexpected(scalars.error());

    if (embeddings->size() != scalars->size()) {
        return std::unexpected(IndexError{std::format(
            "embeddings ({} rows) and scalars ({} rows) disagree",
            embeddings->size(), scalars->size())});
    }

    if (cfg_.metric == MetricType::Cosine) {
        for (auto& emb : *embeddings) {
            l2_normalise(emb);
        }
    }

    auto mm = extract_min_max(*scalars);

    // Build per-object structs; quantise + SFC-encode each in parallel.
    const std::size_t n = embeddings->size();
    std::vector<std::shared_ptr<Object>> objects(n);
    Result<void> first_error{};

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        auto obj = std::make_shared<Object>();
        obj->id = static_cast<std::int32_t>(i);
        obj->embedding = std::move((*embeddings)[static_cast<std::size_t>(i)]);
        obj->scalar_attributes = std::move((*scalars)[static_cast<std::size_t>(i)]);
        obj->norm_sq = norm_sq(obj->embedding);

        auto q = quantize_scalar_attributes(
            obj->scalar_attributes, mm.mins, mm.maxs, cfg_.scalar_bits_per_dimension);
        if (!q) {
            #pragma omp critical
            { if (first_error.has_value()) first_error = std::unexpected(q.error()); }
            continue;
        }
        try {
            obj->sfc_value = sfc_->encode(*q);
        } catch (const std::exception& e) {
            #pragma omp critical
            { if (first_error.has_value())
                first_error = std::unexpected(IndexError{std::format(
                    "SFC encode at row {}: {}", i, e.what())}); }
            continue;
        }
        objects[static_cast<std::size_t>(i)] = std::move(obj);
    }
    if (!first_error.has_value()) return std::unexpected(first_error.error());

    auto split = compute_quantile(objects, cfg_.partitions);
    if (!split) return std::unexpected(split.error());

    // One task per partition. Nested parallelism is disabled so that FAISS
    // k-means does not spawn threads inside an already-parallel region.
    omp_set_max_active_levels(1);
    if (cfg_.threads > 0) omp_set_num_threads(cfg_.threads);

    Result<void> partition_error{};
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(split->partitions.size()); ++i) {
        auto& part = split->partitions[static_cast<std::size_t>(i)];
        auto& part_objs = split->objects_per_partition[static_cast<std::size_t>(i)];
        if (static_cast<int>(part_objs.size()) < cfg_.partition_threshold) {
            build_flat_partition(part, part_objs);
        } else {
            auto r = build_kmeans_partition(part, part_objs, cfg_.metric, cfg_.seed);
            if (!r) {
                #pragma omp critical
                { if (partition_error.has_value())
                    partition_error = std::unexpected(IndexError{std::format(
                        "partition {}: {}", static_cast<int>(i), r.error().message)}); }
            }
        }
    }
    if (!partition_error.has_value()) return std::unexpected(partition_error.error());

    Index idx;
    idx.partitions = std::move(split->partitions);
    idx.min_scalar_attributes = std::move(mm.mins);
    idx.max_scalar_attributes = std::move(mm.maxs);
    return idx;
}

Result<SmartIVF::SearchResultBatch>
SmartIVF::search(const Index& index,
                 std::span<const float> query_vector,
                 std::span<const std::array<float, 2>> raw_filter_ranges,
                 int num_neighbors,
                 std::size_t sfc_budget,
                 int min_probe_bound,
                 int max_probe_bound,
                 int candidate_target) const {
    if (raw_filter_ranges.empty()) {
        return std::unexpected(IndexError{"filter ranges cannot be empty"});
    }

    // Normalise the query if cosine; keep it on the stack (small) so the
    // caller's span isn't mutated.
    std::vector<float> q_storage;
    std::span<const float> q = query_vector;
    if (cfg_.metric == MetricType::Cosine) {
        q_storage.assign(query_vector.begin(), query_vector.end());
        l2_normalise(q_storage);
        q = q_storage;
    }
    const float q_norm_sq = (cfg_.metric == MetricType::L2) ? norm_sq(q) : 1.0f;

    // 1. Normalise filter ranges into [0, 1] using the build-time global bounds.
    std::vector<std::array<float, 2>> normalised(raw_filter_ranges.size());
    for (std::size_t d = 0; d < raw_filter_ranges.size(); ++d) {
        const float lo_attr = index.min_scalar_attributes[d];
        const float hi_attr = index.max_scalar_attributes[d];
        const float range = hi_attr - lo_attr;
        if (range <= 0.0f) {
            normalised[d] = {0.0f, 0.0f};
            continue;
        }
        float lo = (raw_filter_ranges[d][0] - lo_attr) / range;
        float hi = (raw_filter_ranges[d][1] - lo_attr) / range;
        lo = std::clamp(lo, 0.0f, 1.0f);
        hi = std::clamp(hi, 0.0f, 1.0f);
        normalised[d] = {lo, hi};
    }

    // 2. Range decomposition → SFC intervals.
    auto intervals = range_decompose(normalised, *sfc_,
                                     cfg_.scalar_bits_per_dimension, sfc_budget);
    if (!intervals) return std::unexpected(intervals.error());

    // 3. Partition selection: pick every partition whose [min_sfc, max_sfc] hits
    // any interval. Partitions are scanned in index order, so the result is
    // already sorted and duplicate-free without a set.
    std::vector<int> selected;
    selected.reserve(index.partitions.size());
    for (std::size_t i = 0; i < index.partitions.size(); ++i) {
        const auto& p = index.partitions[i];
        for (const auto& iv : *intervals) {
            const std::uint64_t lo = std::max(p.min_sfc, iv.start);
            const std::uint64_t hi = std::min(p.max_sfc, iv.end);
            if (lo <= hi) {
                selected.push_back(static_cast<int>(i));
                break;
            }
        }
    }

    // 4. Safe-adaptive probing.
    std::vector<std::shared_ptr<Object>> pool;
    int dco = 0;
    std::vector<std::shared_ptr<Object>> slice;
    slice.reserve(256);

    for (int pid : selected) {
        const auto& p = index.partitions[static_cast<std::size_t>(pid)];

        // Skip partitions whose scalar bounds miss the query.
        bool ok = true;
        for (std::size_t d = 0; d < raw_filter_ranges.size(); ++d) {
            if (d < p.max_scalar_partition_attributes.size() &&
                d < p.min_scalar_partition_attributes.size()) {
                if (p.max_scalar_partition_attributes[d] < raw_filter_ranges[d][0] ||
                    p.min_scalar_partition_attributes[d] > raw_filter_ranges[d][1]) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) continue;

        if (p.type == PartitionType::Flat) {
            slice.clear();
            collect_by_binary_search(p.objects, *intervals, slice);
            for (const auto& obj : slice) {
                if (scalar_matches(obj->scalar_attributes, raw_filter_ranges)) {
                    pool.push_back(obj);
                }
            }
        } else {  // KMeans
            // Centroid distances (use IP for cosine, precomputed-norm L2 otherwise).
            struct CDist {
                int id;
                float dist;
            };
            std::vector<CDist> cd;
            cd.reserve(p.centroids.size());
            for (std::size_t c = 0; c < p.centroids.size(); ++c) {
                const auto& centroid = p.centroids[c];
                const float d = (cfg_.metric == MetricType::Cosine)
                    ? (1.0f - ip(q, centroid))
                    : l2_with_norms(q, centroid, q_norm_sq, p.centroid_norm_sq[c]);
                cd.push_back({static_cast<int>(c), d});
            }
            dco += static_cast<int>(p.centroids.size());
            std::ranges::sort(cd, {}, &CDist::dist);

            int probes = 0;
            int cands = 0;
            for (const auto& [cid, _] : cd) {
                if (probes >= max_probe_bound) break;
                if (cands >= candidate_target && probes >= min_probe_bound) break;

                auto it = p.inverted_lists.find(cid);
                if (it == p.inverted_lists.end()) continue;
                const auto& list = it->second;

                slice.clear();
                collect_by_binary_search(list, *intervals, slice);
                if (slice.empty()) continue;

                ++probes;
                int survivors = 0;
                for (const auto& obj : slice) {
                    if (scalar_matches(obj->scalar_attributes, raw_filter_ranges)) {
                        pool.push_back(obj);
                        ++survivors;
                    }
                }
                cands += survivors;
            }
        }
    }

    // 5. Refinement: compute the full distance for every candidate.
    struct Hit {
        std::shared_ptr<Object> obj;
        float dist;
    };
    std::vector<Hit> hits;
    hits.reserve(pool.size());
    for (const auto& obj : pool) {
        const float d = (cfg_.metric == MetricType::Cosine)
            ? (1.0f - ip(q, obj->embedding))
            : l2_with_norms(q, obj->embedding, q_norm_sq, obj->norm_sq);
        hits.push_back({obj, d});
    }
    dco += static_cast<int>(pool.size());
    std::ranges::sort(hits, {}, &Hit::dist);

    SearchResultBatch out;
    const int take = std::min(num_neighbors, static_cast<int>(hits.size()));
    out.results.reserve(static_cast<std::size_t>(take));
    for (int i = 0; i < take; ++i) {
        out.results.push_back(std::move(hits[static_cast<std::size_t>(i)].obj));
    }
    out.dco = dco;
    return out;
}

}  // namespace smartivf
