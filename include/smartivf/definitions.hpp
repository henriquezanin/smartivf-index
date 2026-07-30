// Core type definitions for the SmartIVF index.

#pragma once

#include <cstdint>
#include <cstddef>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace smartivf {

// ---- Configuration enums ------------------------------------------------------

enum class MetricType : std::uint8_t {
    L2 = 0,
    Cosine = 1,
};

constexpr std::string_view to_string(MetricType m) noexcept {
    return m == MetricType::Cosine ? "cosine" : "l2";
}

enum class PartitionType : std::uint8_t {
    Flat = 0,
    KMeans = 1,
};

enum class SFCType : std::uint8_t {
    ZOrder = 0,
};

constexpr std::string_view to_string(SFCType s) noexcept {
    return s == SFCType::ZOrder ? "z-order" : "unknown";
}

// ---- Errors -------------------------------------------------------------------

struct IndexError {
    std::string message;
};

template <class T>
using Result = std::expected<T, IndexError>;

// ---- Object: one base vector + scalars + SFC value ---------------------------
//
// IDs are the row index of the object in the base file, so int32 covers the
// dataset sizes reported in the paper and matches the .ivecs ground-truth type.

struct Object {
    std::int32_t id{};
    std::vector<float> embedding;             // length = embedding_dim
    std::vector<float> scalar_attributes;     // length = scalar_dims
    std::uint64_t sfc_value{};
    float norm_sq{};                          // ‖embedding‖² — precomputed (L2 fast path)
};

// ---- Partition ----------------------------------------------------------------
//
// A partition holds either a flat list of objects (sorted by SFC value) or a
// k-means clustering (centroids + inverted lists, each list sorted by SFC).

struct Partition {
    PartitionType type{PartitionType::Flat};

    // Scalar / SFC bounds — populated by compute_quantile, used to skip
    // partitions whose extents don't intersect the query.
    std::vector<float> min_scalar_partition_attributes;
    std::vector<float> max_scalar_partition_attributes;
    std::uint64_t min_sfc{};
    std::uint64_t max_sfc{};

    // FlatPartition state: object list sorted ascending by sfc_value.
    std::vector<std::shared_ptr<Object>> objects;

    // KMeansPartition state:
    //   centroids[c] is the centroid embedding (already normalised for cosine).
    //   centroid_norm_sq[c] = ‖centroids[c]‖² (L2 fast path).
    //   inverted_lists[c] is the list of objects assigned to centroid c,
    //     sorted by sfc_value.
    std::vector<std::vector<float>> centroids;
    std::vector<float> centroid_norm_sq;
    std::map<int, std::vector<std::shared_ptr<Object>>> inverted_lists;
};

// ---- Index --------------------------------------------------------------------

struct Index {
    std::vector<Partition> partitions;
    // Global per-attribute min/max — used to (re)quantise queries at search time.
    std::vector<float> min_scalar_attributes;
    std::vector<float> max_scalar_attributes;
};

// ---- Build / search configuration --------------------------------------------

struct SmartIVFConfig {
    SFCType sfc{SFCType::ZOrder};
    int scalar_dimensions{4};
    std::uint8_t scalar_bits_per_dimension{8};
    int partition_threshold{1000};
    int partitions{50};
    MetricType metric{MetricType::L2};
    std::uint64_t seed{42};      // fed to FAISS k-means for reproducibility
    int threads{0};              // 0 → omp_get_max_threads(); 1 → single-threaded
};

// ---- Search result ------------------------------------------------------------

struct SearchResult {
    std::int32_t id{};
    float distance{};
};

// ---- SFC interval (used by range_decompose) -----------------------------------

struct SFCInterval {
    std::uint64_t start{};
    std::uint64_t end{};
};

}  // namespace smartivf
