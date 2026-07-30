// Per-partition build.
//
// FlatPartition: sort by sfc_value, store as-is.
// KMeansPartition: k-means++ via faiss::Clustering. Centroids are renormalised
//   after k-means for the cosine metric so search-time scoring stays consistent
//   with the normalised base embeddings.
//
// Cluster count rule: k = clamp(floor(sqrt(n)), 2, n)

#pragma once

#include "smartivf/definitions.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace smartivf {

void build_flat_partition(Partition& p,
                          std::vector<std::shared_ptr<Object>>& objects);

[[nodiscard]] Result<void>
build_kmeans_partition(Partition& p,
                       std::vector<std::shared_ptr<Object>>& objects,
                       MetricType metric,
                       std::uint64_t seed);

}  // namespace smartivf
