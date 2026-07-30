#include "smartivf/build_partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <ranges>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

namespace smartivf {

namespace {

[[nodiscard]] int compute_num_clusters(std::size_t n) noexcept {
    int k = static_cast<int>(std::floor(std::sqrt(static_cast<double>(n))));
    if (k < 2) k = 2;
    if (static_cast<std::size_t>(k) > n) k = static_cast<int>(n);
    return k;
}

[[nodiscard]] float norm_sq(std::span<const float> v) noexcept {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
    return static_cast<float>(s);
}

void l2_normalise(std::span<float> v) noexcept {
    const double n = std::sqrt(static_cast<double>(norm_sq(v)));
    if (n > 0.0) {
        const float inv = static_cast<float>(1.0 / n);
        for (auto& x : v) x *= inv;
    }
}

}  // namespace

void build_flat_partition(Partition& p,
                          std::vector<std::shared_ptr<Object>>& objects) {
    std::ranges::sort(objects, {}, [](const auto& o) { return o->sfc_value; });
    p.type = PartitionType::Flat;
    p.objects = std::move(objects);
}

Result<void>
build_kmeans_partition(Partition& p,
                       std::vector<std::shared_ptr<Object>>& objects,
                       MetricType metric,
                       std::uint64_t seed) {
    const std::size_t n = objects.size();
    if (n == 0) {
        return std::unexpected(IndexError{"cannot build kmeans partition on empty object slice"});
    }
    const int k = compute_num_clusters(n);
    const std::size_t dim = objects.front()->embedding.size();

    // Pack embeddings into a contiguous row-major buffer for FAISS.
    std::vector<float> training_data(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& emb = objects[i]->embedding;
        std::copy(emb.begin(), emb.end(),
                  training_data.begin() + static_cast<std::ptrdiff_t>(i * dim));
    }

    // Clustering parameters. `niter = 25` is the FAISS default. The seed is
    // taken from the caller so that runs are reproducible.
    faiss::ClusteringParameters cp;
    cp.niter = 25;
    cp.seed = static_cast<int>(seed);
    cp.spherical = (metric == MetricType::Cosine);
    cp.verbose = false;

    faiss::Clustering clus(static_cast<int>(dim), k, cp);

    if (metric == MetricType::Cosine) {
        faiss::IndexFlatIP coarse(static_cast<int>(dim));
        clus.train(static_cast<faiss::idx_t>(n), training_data.data(), coarse);
    } else {
        faiss::IndexFlatL2 coarse(static_cast<int>(dim));
        clus.train(static_cast<faiss::idx_t>(n), training_data.data(), coarse);
    }

    // Assign each object to nearest centroid via a freshly built flat index.
    std::vector<faiss::idx_t> assignments(n);
    std::vector<float> distances(n);
    if (metric == MetricType::Cosine) {
        faiss::IndexFlatIP assign_idx(static_cast<int>(dim));
        assign_idx.add(static_cast<faiss::idx_t>(k), clus.centroids.data());
        assign_idx.search(static_cast<faiss::idx_t>(n), training_data.data(),
                          1, distances.data(), assignments.data());
    } else {
        faiss::IndexFlatL2 assign_idx(static_cast<int>(dim));
        assign_idx.add(static_cast<faiss::idx_t>(k), clus.centroids.data());
        assign_idx.search(static_cast<faiss::idx_t>(n), training_data.data(),
                          1, distances.data(), assignments.data());
    }

    // Build inverted lists and sort each by sfc_value.
    std::map<int, std::vector<std::shared_ptr<Object>>> inverted;
    for (std::size_t i = 0; i < n; ++i) {
        const int cid = static_cast<int>(assignments[i]);
        inverted[cid].push_back(objects[i]);
    }
    for (auto& [_, lst] : inverted) {
        std::ranges::sort(lst, {}, [](const auto& o) { return o->sfc_value; });
    }

    // Materialise centroids in [0..k) order, taking from FAISS's row-major buffer.
    std::vector<std::vector<float>> centroids(static_cast<std::size_t>(k));
    for (int c = 0; c < k; ++c) {
        centroids[static_cast<std::size_t>(c)].assign(
            clus.centroids.begin() + c * static_cast<int>(dim),
            clus.centroids.begin() + (c + 1) * static_cast<int>(dim));
        if (metric == MetricType::Cosine) {
            l2_normalise(centroids[static_cast<std::size_t>(c)]);
        }
    }

    // Centroid squared norms, used by the L2 path. Unread for cosine.
    std::vector<float> centroid_norm_sq(static_cast<std::size_t>(k));
    for (int c = 0; c < k; ++c) {
        centroid_norm_sq[static_cast<std::size_t>(c)] =
            norm_sq(centroids[static_cast<std::size_t>(c)]);
    }

    p.type = PartitionType::KMeans;
    p.centroids = std::move(centroids);
    p.centroid_norm_sq = std::move(centroid_norm_sq);
    p.inverted_lists = std::move(inverted);
    return {};
}

}  // namespace smartivf
