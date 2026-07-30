// Algorithm 3: SFC range decomposition with BFS + interval merge.

#pragma once

#include "smartivf/definitions.hpp"
#include "smartivf/sfc/sfc.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace smartivf {

// `filter_ranges` is normalised to [0, 1] per dimension (caller responsibility).
// Returns the list of (start, end) SFC intervals — inclusive on both ends —
// whose union covers the multi-dimensional filter region, with at most `budget`
// intervals.
//
// Budget handling: the BFS accepts the current node as one interval as soon as
//   intervals.size() + queue.size() + 1 >= budget,
// so the budget is never exceeded by more than one before forced merging.
[[nodiscard]] Result<std::vector<SFCInterval>>
range_decompose(std::span<const std::array<float, 2>> filter_ranges,
                const sfc::SFC& curve,
                std::uint8_t bits,
                std::size_t budget);

}  // namespace smartivf
