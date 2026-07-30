// CSV query-range reader.
//
// Plain CSV, no header by default. One row per query, even column count.
// Columns are interleaved (min, max) pairs, one pair per scalar dimension.

#pragma once

#include "smartivf/definitions.hpp"

#include <array>
#include <string>
#include <vector>

namespace smartivf::utils {

// Each inner vector has length D; each element is the (min, max) pair for that dim.
using QueryRange = std::vector<std::array<float, 2>>;

[[nodiscard]] Result<std::vector<QueryRange>>
read_ranges_csv(const std::string& path, bool has_header = false);

}  // namespace smartivf::utils
