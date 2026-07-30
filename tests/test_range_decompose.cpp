#include "smartivf/range_decompose.hpp"
#include "smartivf/sfc/z_order.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace smartivf;

TEST(RangeDecompose, FullDomainProducesOneInterval) {
    sfc::ZOrder z(2, 8);
    std::vector<std::array<float, 2>> rng = {{0.0f, 1.0f}, {0.0f, 1.0f}};
    auto r = range_decompose(rng, z, 8, /*budget=*/256);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 1U);
    EXPECT_EQ((*r)[0].start, 0U);
    // Full domain end: encode([255, 255]) = 0xFFFF (16 bits set)
    EXPECT_EQ((*r)[0].end, 0xFFFFULL);
}

TEST(RangeDecompose, EmptyFilterReturnsEmpty) {
    sfc::ZOrder z(2, 8);
    std::vector<std::array<float, 2>> rng;
    auto r = range_decompose(rng, z, 8, 256);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_TRUE(r->empty());
}

TEST(RangeDecompose, BudgetCutoffEnforcesUpperBound) {
    sfc::ZOrder z(3, 8);  // 24-bit codes
    // Narrow, unaligned range in all 3 dims, to force many BFS splits.
    std::vector<std::array<float, 2>> rng = {
        {0.123f, 0.234f}, {0.345f, 0.456f}, {0.567f, 0.678f}};
    for (std::size_t budget : {1U, 4U, 16U, 64U, 256U}) {
        auto r = range_decompose(rng, z, 8, budget);
        ASSERT_TRUE(r.has_value()) << "budget=" << budget << ": " << r.error().message;
        EXPECT_LE(r->size(), budget) << "budget=" << budget;
        // Intervals should be strictly increasing and non-overlapping after merge.
        for (std::size_t i = 1; i < r->size(); ++i) {
            EXPECT_LT((*r)[i - 1].end, (*r)[i].start)
                << "intervals[" << (i - 1) << "].end >= intervals[" << i << "].start";
        }
    }
}

TEST(RangeDecompose, MergeContiguousIntervals) {
    sfc::ZOrder z(2, 4);
    // Single dimension, contiguous 0..1 range covers entire 4-bit row → one interval
    std::vector<std::array<float, 2>> rng = {{0.0f, 1.0f}, {0.0f, 1.0f}};
    auto r = range_decompose(rng, z, 4, 256);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), 1U);
}
