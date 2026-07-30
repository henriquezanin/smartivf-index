// Z-order (Morton) SFC encoder.
//
// Bit layout: for D dimensions and B bits per dimension, the output bit at
// position (b * D + d) holds bit b of coord d (LSB-first interleaving).
//
// Two interchangeable implementations, selected at runtime:
//   - BMI2 path (`_pdep_u64` / `_pext_u64`), used when
//     `__builtin_cpu_supports("bmi2")` returns true.
//   - Scalar fallback: per-bit shift+OR. Same output, portable to non-BMI2 CPUs.

#pragma once

#include "smartivf/sfc/sfc.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace smartivf::sfc {

class ZOrder final : public SFC {
public:
    // Throws std::invalid_argument if dims == 0, bits == 0, or dims*bits > 64.
    ZOrder(std::size_t dims, std::uint8_t bits);

    [[nodiscard]] std::size_t dimensions() const noexcept override { return dims_; }
    [[nodiscard]] std::uint8_t bits_per_dimension() const noexcept override { return bits_; }

    [[nodiscard]] std::uint64_t encode(std::span<const std::uint64_t> coords) const override;
    void decode(std::uint64_t code, std::span<std::uint64_t> out) const override;

    // Public for tests / direct use without virtual dispatch.
    [[nodiscard]] std::uint64_t mask_for_dim(std::size_t d) const noexcept { return masks_[d]; }

private:
    std::size_t dims_{};
    std::uint8_t bits_{};
    bool bmi2_{};
    // masks_[d] has bit set at every position {d, d+D, d+2D, …, d+(B-1)D},
    // i.e. the destination positions for dim d in the interleaved code.
    std::array<std::uint64_t, 64> masks_{};

    [[nodiscard]] std::uint64_t encode_bmi2(std::span<const std::uint64_t> coords) const noexcept;
    [[nodiscard]] std::uint64_t encode_scalar(std::span<const std::uint64_t> coords) const noexcept;
    void decode_bmi2(std::uint64_t code, std::span<std::uint64_t> out) const noexcept;
    void decode_scalar(std::uint64_t code, std::span<std::uint64_t> out) const noexcept;
};

}  // namespace smartivf::sfc
