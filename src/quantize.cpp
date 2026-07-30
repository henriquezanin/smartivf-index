#include "smartivf/quantize.hpp"

#include <algorithm>
#include <cstdint>
#include <format>

namespace smartivf {

Result<std::vector<std::uint64_t>>
quantize_scalar_attributes(std::span<const float> scalar_attributes,
                           std::span<const float> mins,
                           std::span<const float> maxs,
                           std::uint8_t bits_per_dimension) {
    if (scalar_attributes.size() != mins.size() ||
        scalar_attributes.size() != maxs.size()) {
        return std::unexpected(IndexError{std::format(
            "dimension mismatch: attrs={}, mins={}, maxs={}",
            scalar_attributes.size(), mins.size(), maxs.size())});
    }
    const double max_quantized = static_cast<double>(
        (std::uint64_t{1} << bits_per_dimension) - 1);
    std::vector<std::uint64_t> out(scalar_attributes.size());
    for (std::size_t i = 0; i < scalar_attributes.size(); ++i) {
        const double range = static_cast<double>(maxs[i] - mins[i]);
        if (range <= 0.0) {
            out[i] = 0;
            continue;
        }
        double normalised =
            static_cast<double>(scalar_attributes[i] - mins[i]) / range;
        normalised = std::clamp(normalised, 0.0, 1.0);
        out[i] = static_cast<std::uint64_t>(normalised * max_quantized);
    }
    return out;
}

MinMax extract_min_max(std::span<const std::vector<float>> scalar_attributes) {
    if (scalar_attributes.empty()) return {};
    const std::size_t dim = scalar_attributes[0].size();
    MinMax out{std::vector<float>(dim), std::vector<float>(dim)};
    for (std::size_t d = 0; d < dim; ++d) {
        out.mins[d] = scalar_attributes[0][d];
        out.maxs[d] = scalar_attributes[0][d];
    }
    for (const auto& row : scalar_attributes) {
        for (std::size_t d = 0; d < dim; ++d) {
            if (row[d] < out.mins[d]) out.mins[d] = row[d];
            if (row[d] > out.maxs[d]) out.maxs[d] = row[d];
        }
    }
    return out;
}

}  // namespace smartivf
