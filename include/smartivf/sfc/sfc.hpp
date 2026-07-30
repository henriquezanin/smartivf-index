// Abstract space-filling-curve interface. Concrete implementations live in
// dedicated headers.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace smartivf::sfc {

// An SFC maps a D-dim point with B bits/dim into a single uint64 code.
// Constraint enforced by every implementation: dims * bits ≤ 64.
class SFC {
public:
    virtual ~SFC() = default;

    [[nodiscard]] virtual std::size_t dimensions() const noexcept = 0;
    [[nodiscard]] virtual std::uint8_t bits_per_dimension() const noexcept = 0;

    // Encode the D coordinates into a single code.
    // Pre-condition: coords.size() == dimensions(); each coord fits in `bits_per_dimension()` bits.
    [[nodiscard]] virtual std::uint64_t encode(std::span<const std::uint64_t> coords) const = 0;

    // Decode a single code back into D coordinates.
    // Writes exactly dimensions() values into `out` (must be sized accordingly).
    virtual void decode(std::uint64_t code, std::span<std::uint64_t> out) const = 0;
};

}  // namespace smartivf::sfc
