#include "smartivf/compute_quantile.hpp"

#include <algorithm>
#include <format>
#include <ranges>
#include <utility>

namespace smartivf {

Result<QuantileSplit>
compute_quantile(std::vector<std::shared_ptr<Object>>& objects, int partitions) {
    const std::size_t n = objects.size();
    if (n == 0) {
        return std::unexpected(IndexError{"cannot compute quantiles on empty object slice"});
    }
    if (partitions <= 0) {
        return std::unexpected(IndexError{std::format(
            "partitions must be positive, got {}", partitions)});
    }
    if (static_cast<std::size_t>(partitions) > n) {
        return std::unexpected(IndexError{std::format(
            "partitions ({}) exceeds number of objects ({})", partitions, n)});
    }

    std::ranges::sort(objects, {}, [](const auto& o) { return o->sfc_value; });

    QuantileSplit out;
    out.partitions.resize(static_cast<std::size_t>(partitions));
    out.objects_per_partition.resize(static_cast<std::size_t>(partitions));

    for (std::size_t i = 0; i < static_cast<std::size_t>(partitions); ++i) {
        const std::size_t start = (i * n) / static_cast<std::size_t>(partitions);
        const std::size_t end   = ((i + 1) * n) / static_cast<std::size_t>(partitions);

        // Slice into a fresh vector of shared_ptr — cheap shallow copy since the
        // objects themselves are owned by shared_ptr.
        std::vector<std::shared_ptr<Object>> partition_objs(
            objects.begin() + static_cast<std::ptrdiff_t>(start),
            objects.begin() + static_cast<std::ptrdiff_t>(end));

        const std::size_t dims = partition_objs.front()->scalar_attributes.size();
        Partition p;
        p.min_scalar_partition_attributes.assign(dims, 0);
        p.max_scalar_partition_attributes.assign(dims, 0);
        for (std::size_t d = 0; d < dims; ++d) {
            p.min_scalar_partition_attributes[d] = partition_objs.front()->scalar_attributes[d];
            p.max_scalar_partition_attributes[d] = partition_objs.front()->scalar_attributes[d];
        }
        for (const auto& o : partition_objs) {
            for (std::size_t d = 0; d < dims; ++d) {
                const float v = o->scalar_attributes[d];
                if (v < p.min_scalar_partition_attributes[d]) p.min_scalar_partition_attributes[d] = v;
                if (v > p.max_scalar_partition_attributes[d]) p.max_scalar_partition_attributes[d] = v;
            }
        }
        p.min_sfc = partition_objs.front()->sfc_value;
        p.max_sfc = partition_objs.back()->sfc_value;

        out.partitions[i] = std::move(p);
        out.objects_per_partition[i] = std::move(partition_objs);
    }
    return out;
}

}  // namespace smartivf
