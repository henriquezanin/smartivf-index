// Binary index serialisation: a little-endian format with a magic+version
// header, so indices written by an incompatible build are rejected cleanly.

#pragma once

#include "smartivf/definitions.hpp"

#include <string>

namespace smartivf {

inline constexpr std::uint32_t kIndexFileMagic   = 0x53494643u;   // 'SIFC'
inline constexpr std::uint32_t kIndexFileVersion = 1u;

[[nodiscard]] Result<void> store_index(const Index& idx, const std::string& path);
[[nodiscard]] Result<Index> load_index(const std::string& path);

}  // namespace smartivf
