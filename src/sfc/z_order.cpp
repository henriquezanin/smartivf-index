#include "smartivf/sfc/z_order.hpp"

#include <format>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

namespace smartivf::sfc {

namespace {

[[nodiscard]] bool detect_bmi2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

}  // namespace

ZOrder::ZOrder(std::size_t dims, std::uint8_t bits)
    : dims_(dims), bits_(bits), bmi2_(detect_bmi2()) {
    if (dims_ == 0) {
        throw std::invalid_argument("ZOrder: dimensions must be > 0");
    }
    if (bits_ == 0) {
        throw std::invalid_argument("ZOrder: bits_per_dimension must be > 0");
    }
    if (dims_ * static_cast<std::size_t>(bits_) > 64) {
        throw std::invalid_argument(std::format(
            "ZOrder: dimensions ({}) × bits ({}) = {} exceeds 64",
            dims_, bits_, dims_ * static_cast<std::size_t>(bits_)));
    }
    // Build per-dim deposit/extract masks: bit set at every interleaved position
    // that dim d owns in the output.
    for (std::size_t d = 0; d < dims_; ++d) {
        std::uint64_t m = 0;
        for (std::uint8_t b = 0; b < bits_; ++b) {
            m |= std::uint64_t{1} << (b * dims_ + d);
        }
        masks_[d] = m;
    }
}

std::uint64_t ZOrder::encode(std::span<const std::uint64_t> coords) const {
    if (coords.size() != dims_) {
        throw std::invalid_argument(std::format(
            "ZOrder::encode: expected {} coords, got {}", dims_, coords.size()));
    }
    const std::uint64_t max_coord = (bits_ == 64) ? ~std::uint64_t{0}
                                                  : (std::uint64_t{1} << bits_) - 1;
    for (std::size_t d = 0; d < dims_; ++d) {
        if (coords[d] > max_coord) {
            throw std::invalid_argument(std::format(
                "ZOrder::encode: coord[{}] = {} does not fit in {} bits",
                d, coords[d], bits_));
        }
    }
    return bmi2_ ? encode_bmi2(coords) : encode_scalar(coords);
}

void ZOrder::decode(std::uint64_t code, std::span<std::uint64_t> out) const {
    if (out.size() != dims_) {
        throw std::invalid_argument(std::format(
            "ZOrder::decode: out span must hold {} coords, got {}", dims_, out.size()));
    }
    if (bmi2_) {
        decode_bmi2(code, out);
    } else {
        decode_scalar(code, out);
    }
}

std::uint64_t ZOrder::encode_bmi2(std::span<const std::uint64_t> coords) const noexcept {
#if defined(__BMI2__)
    std::uint64_t code = 0;
    for (std::size_t d = 0; d < dims_; ++d) {
        code |= _pdep_u64(coords[d], masks_[d]);
    }
    return code;
#else
    return encode_scalar(coords);
#endif
}

std::uint64_t ZOrder::encode_scalar(std::span<const std::uint64_t> coords) const noexcept {
    std::uint64_t code = 0;
    for (std::uint8_t b = 0; b < bits_; ++b) {
        for (std::size_t d = 0; d < dims_; ++d) {
            const std::uint64_t bit = (coords[d] >> b) & 1ULL;
            code |= bit << (b * dims_ + d);
        }
    }
    return code;
}

void ZOrder::decode_bmi2(std::uint64_t code, std::span<std::uint64_t> out) const noexcept {
#if defined(__BMI2__)
    for (std::size_t d = 0; d < dims_; ++d) {
        out[d] = _pext_u64(code, masks_[d]);
    }
#else
    decode_scalar(code, out);
#endif
}

void ZOrder::decode_scalar(std::uint64_t code, std::span<std::uint64_t> out) const noexcept {
    for (std::size_t d = 0; d < dims_; ++d) {
        out[d] = 0;
    }
    for (std::uint8_t b = 0; b < bits_; ++b) {
        for (std::size_t d = 0; d < dims_; ++d) {
            const std::uint64_t bit = (code >> (b * dims_ + d)) & 1ULL;
            out[d] |= bit << b;
        }
    }
}

}  // namespace smartivf::sfc
