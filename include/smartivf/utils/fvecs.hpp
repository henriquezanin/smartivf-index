// fvecs / ivecs reader.
//
// Binary layout per row:
//   - 4 bytes: little-endian int32 dim
//   - dim * 4 bytes: little-endian float32 / int32 values
// All rows must share the same dim — this is the ANN-benchmarks convention.

#pragma once

#include "smartivf/definitions.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace smartivf::utils {

[[nodiscard]] Result<std::vector<std::vector<float>>>
read_fvecs(const std::string& path);

[[nodiscard]] Result<std::vector<std::vector<std::int32_t>>>
read_ivecs(const std::string& path);

}  // namespace smartivf::utils
