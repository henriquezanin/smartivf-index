// Base-scalar CSV reader.
//
// Plain CSV, no header by default. One row per base vector, D columns per row.

#pragma once

#include "smartivf/definitions.hpp"

#include <string>
#include <vector>

namespace smartivf::utils {

[[nodiscard]] Result<std::vector<std::vector<float>>>
read_scalar_csv(const std::string& path, bool has_header = false);

}  // namespace smartivf::utils
