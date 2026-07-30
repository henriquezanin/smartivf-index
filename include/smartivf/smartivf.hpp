// SmartIVF: Algorithms 1 (BuildIndex) and 2 (Safe-Adaptive Search).

#pragma once

#include "smartivf/definitions.hpp"
#include "smartivf/sfc/sfc.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace smartivf {

class SmartIVF {
public:
    explicit SmartIVF(const SmartIVFConfig& cfg);

    // Build an index from a base-vectors .fvecs file and a per-row scalars CSV.
    [[nodiscard]] Result<Index>
    build_index(const std::string& fvecs_file,
                const std::string& scalars_file) const;

    // Search a pre-built index. Returns (results, distance_computations).
    struct SearchResultBatch {
        std::vector<std::shared_ptr<Object>> results;
        int dco{};
    };
    [[nodiscard]] Result<SearchResultBatch>
    search(const Index& index,
           std::span<const float> query_vector,
           std::span<const std::array<float, 2>> raw_filter_ranges,
           int num_neighbors,
           std::size_t sfc_budget,
           int min_probe_bound,
           int max_probe_bound,
           int candidate_target) const;

    [[nodiscard]] const sfc::SFC& sfc() const noexcept { return *sfc_; }
    [[nodiscard]] const SmartIVFConfig& config() const noexcept { return cfg_; }

private:
    SmartIVFConfig cfg_{};
    std::unique_ptr<sfc::SFC> sfc_;
};

}  // namespace smartivf
