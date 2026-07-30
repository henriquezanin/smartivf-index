#include "smartivf/utils/fvecs.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <type_traits>
#include <vector>

namespace smartivf::utils {

namespace {

// Read a stream of (dim_header, dim * sizeof(T)) records. T = float or int32_t.
// All rows must share dim. Returns one outer vector per row.
template <class T>
[[nodiscard]] Result<std::vector<std::vector<T>>>
read_vecs(const std::string& path) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::int32_t>);

    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::unexpected(IndexError{std::format(
            "failed to stat {}: {}", path, ec.message())});
    }
    if (file_size < 4) {
        return std::unexpected(IndexError{std::format(
            "{} too small: {} bytes", path, file_size)});
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(IndexError{std::format("failed to open {}", path)});
    }

    std::int32_t dim_hdr{};
    f.read(reinterpret_cast<char*>(&dim_hdr), sizeof(dim_hdr));
    if (!f) {
        return std::unexpected(IndexError{"failed to read dim header"});
    }
    if constexpr (std::endian::native == std::endian::big) {
        dim_hdr = static_cast<std::int32_t>(__builtin_bswap32(static_cast<std::uint32_t>(dim_hdr)));
    }
    if (dim_hdr <= 0) {
        return std::unexpected(IndexError{std::format("invalid dim {}", dim_hdr)});
    }

    const std::uint64_t row_size = 4 + static_cast<std::uint64_t>(dim_hdr) * sizeof(T);
    if (file_size % row_size != 0) {
        return std::unexpected(IndexError{std::format(
            "{}: size {} not a multiple of row size {} (dim={})",
            path, file_size, row_size, dim_hdr)});
    }
    const std::size_t n = static_cast<std::size_t>(file_size / row_size);
    const std::size_t dim = static_cast<std::size_t>(dim_hdr);

    std::vector<std::vector<T>> out;
    out.reserve(n);

    f.clear();
    f.seekg(0, std::ios::beg);
    for (std::size_t i = 0; i < n; ++i) {
        std::int32_t d{};
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!f) {
            return std::unexpected(IndexError{std::format(
                "failed to read dim header at row {} of {}", i, path)});
        }
        if constexpr (std::endian::native == std::endian::big) {
            d = static_cast<std::int32_t>(__builtin_bswap32(static_cast<std::uint32_t>(d)));
        }
        if (d != dim_hdr) {
            return std::unexpected(IndexError{std::format(
                "inconsistent dim at row {}: got {}, want {}", i, d, dim_hdr)});
        }
        std::vector<T> row(dim);
        f.read(reinterpret_cast<char*>(row.data()),
               static_cast<std::streamsize>(dim * sizeof(T)));
        if (!f) {
            return std::unexpected(IndexError{std::format(
                "failed to read row {} of {}", i, path)});
        }
        if constexpr (std::endian::native == std::endian::big) {
            for (auto& v : row) {
                if constexpr (std::is_same_v<T, float>) {
                    std::uint32_t raw = std::bit_cast<std::uint32_t>(v);
                    raw = __builtin_bswap32(raw);
                    v = std::bit_cast<float>(raw);
                } else {
                    v = static_cast<std::int32_t>(
                        __builtin_bswap32(static_cast<std::uint32_t>(v)));
                }
            }
        }
        out.push_back(std::move(row));
    }
    return out;
}

}  // namespace

Result<std::vector<std::vector<float>>> read_fvecs(const std::string& path) {
    return read_vecs<float>(path);
}

Result<std::vector<std::vector<std::int32_t>>> read_ivecs(const std::string& path) {
    return read_vecs<std::int32_t>(path);
}

}  // namespace smartivf::utils
