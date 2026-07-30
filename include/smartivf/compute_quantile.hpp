// Equi-depth partitioning.

#pragma once

#include "smartivf/definitions.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace smartivf {

// Output of compute_quantile:
//   - partitions[i] has its SFC bounds and scalar bounds populated; type and
//     payload (flat objects vs k-means structure) are filled in afterwards by
//     build_partition.
//   - objects_per_partition[i] holds the per-partition object slice (still
//     sorted by SFC value, sliced from the globally sorted input).
struct QuantileSplit {
    std::vector<Partition> partitions;
    std::vector<std::vector<std::shared_ptr<Object>>> objects_per_partition;
};

// Sorts `objects` in place by sfc_value, then splits it into `partitions`
// equi-depth buckets with integer-arithmetic boundaries
// (start = i*n/p, end = (i+1)*n/p).
[[nodiscard]] Result<QuantileSplit>
compute_quantile(std::vector<std::shared_ptr<Object>>& objects, int partitions);

}  // namespace smartivf
