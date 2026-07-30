// Scalar quantisation.

#pragma once

#include "smartivf/definitions.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace smartivf {

// Map scalar attribute values from their original [min, max] ranges to integer
// codes in [0, 2^bits − 1]. Clamping happens after normalisation, so values that
// drift slightly outside the train-time bounds still produce valid codes.
//
// Returns IndexError on dimension mismatch.
[[nodiscard]] Result<std::vector<std::uint64_t>>
quantize_scalar_attributes(std::span<const float> scalar_attributes,
                           std::span<const float> mins,
                           std::span<const float> maxs,
                           std::uint8_t bits_per_dimension);

// Extract per-dimension min and max from a batch of scalar rows.
// Pre-condition: all inner rows share the same length (no validation; caller
// guarantees this from the CSV loader).
struct MinMax {
    std::vector<float> mins;
    std::vector<float> maxs;
};
[[nodiscard]] MinMax extract_min_max(
    std::span<const std::vector<float>> scalar_attributes);

}  // namespace smartivf
