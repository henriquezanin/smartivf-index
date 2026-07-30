#include "smartivf/store.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <memory>
#include <string>

namespace smartivf {

namespace {

// All writes are little-endian; the helpers byte-swap on big-endian targets.
template <class T>
T to_le(T v) noexcept {
    if constexpr (std::endian::native == std::endian::little) {
        return v;
    } else {
        if constexpr (sizeof(T) == 8) return std::bit_cast<T>(__builtin_bswap64(std::bit_cast<std::uint64_t>(v)));
        if constexpr (sizeof(T) == 4) return std::bit_cast<T>(__builtin_bswap32(std::bit_cast<std::uint32_t>(v)));
        if constexpr (sizeof(T) == 2) return std::bit_cast<T>(__builtin_bswap16(std::bit_cast<std::uint16_t>(v)));
        return v;
    }
}

template <class T>
void write_pod(std::ofstream& f, T v) {
    v = to_le(v);
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
[[nodiscard]] T read_pod(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return to_le(v);
}

void write_floats(std::ofstream& f, const std::vector<float>& xs) {
    write_pod<std::uint64_t>(f, xs.size());
    if (xs.empty()) return;
    if constexpr (std::endian::native == std::endian::little) {
        f.write(reinterpret_cast<const char*>(xs.data()),
                static_cast<std::streamsize>(xs.size() * sizeof(float)));
    } else {
        for (float v : xs) write_pod(f, v);
    }
}

[[nodiscard]] std::vector<float> read_floats(std::ifstream& f) {
    const auto n = read_pod<std::uint64_t>(f);
    std::vector<float> out(static_cast<std::size_t>(n));
    if (n == 0) return out;
    if constexpr (std::endian::native == std::endian::little) {
        f.read(reinterpret_cast<char*>(out.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    } else {
        for (auto& v : out) v = read_pod<float>(f);
    }
    return out;
}

void write_object(std::ofstream& f, const Object& o) {
    write_pod<std::int32_t>(f, o.id);
    write_pod<float>(f, o.norm_sq);
    write_pod<std::uint64_t>(f, o.sfc_value);
    write_floats(f, o.embedding);
    write_floats(f, o.scalar_attributes);
}

[[nodiscard]] std::shared_ptr<Object> read_object(std::ifstream& f) {
    auto o = std::make_shared<Object>();
    o->id              = read_pod<std::int32_t>(f);
    o->norm_sq         = read_pod<float>(f);
    o->sfc_value       = read_pod<std::uint64_t>(f);
    o->embedding         = read_floats(f);
    o->scalar_attributes = read_floats(f);
    return o;
}

}  // namespace

Result<void> store_index(const Index& idx, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(IndexError{std::format("failed to create {}", path)});
    }

    write_pod<std::uint32_t>(f, kIndexFileMagic);
    write_pod<std::uint32_t>(f, kIndexFileVersion);

    write_floats(f, idx.min_scalar_attributes);
    write_floats(f, idx.max_scalar_attributes);

    write_pod<std::uint64_t>(f, idx.partitions.size());
    for (const auto& p : idx.partitions) {
        write_pod<std::uint8_t>(f, static_cast<std::uint8_t>(p.type));
        write_pod<std::uint64_t>(f, p.min_sfc);
        write_pod<std::uint64_t>(f, p.max_sfc);
        write_floats(f, p.min_scalar_partition_attributes);
        write_floats(f, p.max_scalar_partition_attributes);

        if (p.type == PartitionType::Flat) {
            write_pod<std::uint64_t>(f, p.objects.size());
            for (const auto& o : p.objects) write_object(f, *o);
        } else {  // KMeans
            write_pod<std::uint64_t>(f, p.centroids.size());
            for (const auto& c : p.centroids)   write_floats(f, c);
            write_floats(f, p.centroid_norm_sq);
            // Inverted lists — write count then (centroid_id, len, [objects]) tuples.
            write_pod<std::uint64_t>(f, p.inverted_lists.size());
            for (const auto& [cid, lst] : p.inverted_lists) {
                write_pod<std::int32_t>(f, cid);
                write_pod<std::uint64_t>(f, lst.size());
                for (const auto& o : lst) write_object(f, *o);
            }
        }
    }
    f.flush();
    if (!f) return std::unexpected(IndexError{"write failed"});
    return {};
}

Result<Index> load_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(IndexError{std::format("failed to open {}", path)});
    }
    const auto magic   = read_pod<std::uint32_t>(f);
    const auto version = read_pod<std::uint32_t>(f);
    if (magic != kIndexFileMagic) {
        return std::unexpected(IndexError{std::format(
            "{}: not a SmartIVF index (magic = 0x{:08x})", path, magic)});
    }
    if (version != kIndexFileVersion) {
        return std::unexpected(IndexError{std::format(
            "{}: unsupported index version {} (this build expects {})",
            path, version, kIndexFileVersion)});
    }

    Index idx;
    idx.min_scalar_attributes = read_floats(f);
    idx.max_scalar_attributes = read_floats(f);

    const auto npart = read_pod<std::uint64_t>(f);
    idx.partitions.resize(static_cast<std::size_t>(npart));
    for (std::size_t i = 0; i < npart; ++i) {
        Partition& p = idx.partitions[i];
        const auto type_byte = read_pod<std::uint8_t>(f);
        p.type = (type_byte == static_cast<std::uint8_t>(PartitionType::KMeans))
                 ? PartitionType::KMeans : PartitionType::Flat;
        p.min_sfc = read_pod<std::uint64_t>(f);
        p.max_sfc = read_pod<std::uint64_t>(f);
        p.min_scalar_partition_attributes = read_floats(f);
        p.max_scalar_partition_attributes = read_floats(f);

        if (p.type == PartitionType::Flat) {
            const auto n = read_pod<std::uint64_t>(f);
            p.objects.reserve(static_cast<std::size_t>(n));
            for (std::uint64_t j = 0; j < n; ++j) p.objects.push_back(read_object(f));
        } else {
            const auto k = read_pod<std::uint64_t>(f);
            p.centroids.resize(static_cast<std::size_t>(k));
            for (auto& c : p.centroids) c = read_floats(f);
            p.centroid_norm_sq = read_floats(f);
            const auto nlists = read_pod<std::uint64_t>(f);
            for (std::uint64_t l = 0; l < nlists; ++l) {
                const auto cid = read_pod<std::int32_t>(f);
                const auto m   = read_pod<std::uint64_t>(f);
                auto& lst = p.inverted_lists[cid];
                lst.reserve(static_cast<std::size_t>(m));
                for (std::uint64_t j = 0; j < m; ++j) lst.push_back(read_object(f));
            }
        }
    }
    if (!f) return std::unexpected(IndexError{"read failed (truncated file?)"});
    return idx;
}

}  // namespace smartivf
