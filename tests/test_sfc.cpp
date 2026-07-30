// Z-order encoder tests: accessors, configuration validation, round-trips and
// fixed known-value tables.

#include "smartivf/sfc/z_order.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace smartivf::sfc;

TEST(ZOrder, Accessors) {
    ZOrder z(3, 10);
    EXPECT_EQ(z.dimensions(), 3U);
    EXPECT_EQ(z.bits_per_dimension(), 10U);
}

TEST(ZOrder, InvalidConfig) {
    EXPECT_THROW(ZOrder(0, 8), std::invalid_argument);
    EXPECT_THROW(ZOrder(2, 0), std::invalid_argument);
    EXPECT_THROW(ZOrder(5, 13), std::invalid_argument);  // 65 bits
    EXPECT_NO_THROW(ZOrder(8, 8));                       // 64 bits boundary OK
    EXPECT_NO_THROW(ZOrder(1, 64));
}

TEST(ZOrder, RoundTrip) {
    struct Case {
        std::size_t dims;
        std::uint8_t bits;
        std::vector<std::uint64_t> coords;
    };
    const std::vector<Case> cases = {
        {2, 4, {0, 0}},
        {2, 4, {3, 5}},
        {2, 4, {15, 15}},
        {2, 8, {100, 200}},
        {3, 4, {1, 2, 3}},
        {3, 4, {15, 15, 15}},
        {3, 8, {50, 100, 200}},
        {2, 16, {1000, 2000}},
        {2, 32, {123456789, 987654321}},
        {1, 64, {std::numeric_limits<std::uint64_t>::max()}},
        {1, 32, {42}},
        {4, 4, {1, 2, 3, 4}},
        {8, 8, {1, 2, 3, 4, 5, 6, 7, 8}},
    };
    for (const auto& c : cases) {
        ZOrder z(c.dims, c.bits);
        std::vector<std::uint64_t> out(c.dims);
        const std::uint64_t code = z.encode(c.coords);
        z.decode(code, out);
        EXPECT_EQ(out, c.coords)
            << "dims=" << c.dims << " bits=" << static_cast<int>(c.bits);
    }
}

// Known 2-D, 3-bit codes. These pin down the LSB-first interleaving.
TEST(ZOrder, KnownValues_2D3Bit) {
    ZOrder z(2, 3);
    struct {
        std::array<std::uint64_t, 2> coords;
        std::uint64_t want;
    } cases[] = {
        {{0, 0}, 0},
        {{1, 0}, 1},
        {{0, 1}, 2},
        {{1, 1}, 3},
        {{7, 7}, 63},
        {{5, 6}, 57},
        {{2, 3}, 14},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(z.encode(c.coords), c.want)
            << "coords=(" << c.coords[0] << "," << c.coords[1] << ")";
    }
}

TEST(ZOrder, KnownValues_2D2Bit) {
    ZOrder z(2, 2);
    struct {
        std::array<std::uint64_t, 2> coords;
        std::uint64_t want;
    } cases[] = {
        {{0, 0},  0}, {{1, 0},  1}, {{2, 0},  4}, {{3, 0},  5},
        {{0, 1},  2}, {{1, 1},  3}, {{2, 1},  6}, {{3, 1},  7},
        {{0, 2},  8}, {{1, 2},  9}, {{2, 2}, 12}, {{3, 2}, 13},
        {{0, 3}, 10}, {{1, 3}, 11}, {{2, 3}, 14}, {{3, 3}, 15},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(z.encode(c.coords), c.want)
            << "coords=(" << c.coords[0] << "," << c.coords[1] << ")";
    }
}

TEST(ZOrder, Encode_WrongCoordCount) {
    ZOrder z(2, 8);
    EXPECT_THROW(z.encode(std::vector<std::uint64_t>{1}), std::invalid_argument);
    EXPECT_THROW(z.encode(std::vector<std::uint64_t>{1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(z.encode(std::vector<std::uint64_t>{}), std::invalid_argument);
}

TEST(ZOrder, Encode_CoordOutOfRange) {
    ZOrder z(2, 4);  // 4 bits → max 15
    EXPECT_THROW(z.encode(std::array<std::uint64_t, 2>{16, 0}), std::invalid_argument);
    EXPECT_THROW(z.encode(std::array<std::uint64_t, 2>{0, 16}), std::invalid_argument);
    EXPECT_THROW(z.encode(std::array<std::uint64_t, 2>{256, 256}), std::invalid_argument);
    EXPECT_THROW(z.encode(std::array<std::uint64_t, 2>{15, 16}), std::invalid_argument);
}

TEST(ZOrder, Boundary_AllZeros) {
    for (std::size_t d : {1U, 2U, 3U, 4U}) {
        ZOrder z(d, 8);
        std::vector<std::uint64_t> zeros(d, 0);
        EXPECT_EQ(z.encode(zeros), 0U);
    }
}

TEST(ZOrder, Boundary_MaxValues) {
    struct { std::size_t dims; std::uint8_t bits; } cases[] = {
        {2, 4}, {2, 16}, {2, 32}, {3, 21}, {4, 16}, {1, 64},
    };
    for (const auto& c : cases) {
        ZOrder z(c.dims, c.bits);
        const std::uint64_t max_val =
            (c.bits >= 64) ? std::numeric_limits<std::uint64_t>::max()
                           : ((std::uint64_t{1} << c.bits) - 1);
        std::vector<std::uint64_t> coords(c.dims, max_val), out(c.dims);
        const auto code = z.encode(coords);
        z.decode(code, out);
        EXPECT_EQ(out, coords)
            << "dims=" << c.dims << " bits=" << static_cast<int>(c.bits);
    }
}

TEST(ZOrder, OneDim_IsIdentity) {
    ZOrder z(1, 16);
    for (std::uint64_t v : {0ULL, 1ULL, 2ULL, 100ULL, 255ULL, 1000ULL, 65535ULL}) {
        EXPECT_EQ(z.encode(std::array<std::uint64_t, 1>{v}), v);
    }
}

// Random round-trip over many (dims, bits) combinations. Covers the BMI2 path on
// supporting CPUs and the scalar path elsewhere.
TEST(ZOrder, RandomRoundTrip) {
    std::mt19937_64 rng(0xC0FFEE);
    for (std::size_t dims : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
        for (std::uint8_t bits : {1, 2, 4, 7, 8, 12, 16}) {
            if (dims * static_cast<std::size_t>(bits) > 64) continue;
            ZOrder z(dims, bits);
            const std::uint64_t max_coord =
                (bits == 64) ? std::numeric_limits<std::uint64_t>::max()
                             : ((std::uint64_t{1} << bits) - 1);

            std::vector<std::uint64_t> coords(dims), out(dims);
            for (int iter = 0; iter < 64; ++iter) {
                for (std::size_t d = 0; d < dims; ++d) {
                    coords[d] = rng() & max_coord;
                }
                z.decode(z.encode(coords), out);
                ASSERT_EQ(out, coords)
                    << "dims=" << dims << " bits=" << static_cast<int>(bits);
            }
        }
    }
}
