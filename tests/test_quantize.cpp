#include "smartivf/quantize.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace smartivf;

TEST(Quantize, NormalisesAndClampsToBitRange) {
    std::vector<float> mins{0.0f, 10.0f};
    std::vector<float> maxs{100.0f, 30.0f};
    // 4 bits → max code 15.
    auto r = quantize_scalar_attributes({{50.0f, 20.0f}}, mins, maxs, 4);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ((*r)[0], 7U);   // 50/100 * 15 = 7.5 → trunc to 7
    EXPECT_EQ((*r)[1], 7U);   // 10/20 * 15 = 7.5 → trunc to 7

    // Below-min clamps to 0; above-max clamps to maxCode.
    auto under = quantize_scalar_attributes({{-50.0f, 0.0f}}, mins, maxs, 4);
    ASSERT_TRUE(under.has_value());
    EXPECT_EQ((*under)[0], 0U);
    EXPECT_EQ((*under)[1], 0U);

    auto over = quantize_scalar_attributes({{500.0f, 100.0f}}, mins, maxs, 4);
    ASSERT_TRUE(over.has_value());
    EXPECT_EQ((*over)[0], 15U);
    EXPECT_EQ((*over)[1], 15U);
}

TEST(Quantize, ZeroRangeProducesZero) {
    std::vector<float> mins{5.0f};
    std::vector<float> maxs{5.0f};
    auto r = quantize_scalar_attributes({{5.0f}}, mins, maxs, 8);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ((*r)[0], 0U);
}

TEST(Quantize, DimensionMismatch) {
    auto r = quantize_scalar_attributes({{1.0f}}, {{0.0f, 0.0f}}, {{1.0f, 1.0f}}, 8);
    EXPECT_FALSE(r.has_value());
}

TEST(Quantize, ExtractMinMax) {
    std::vector<std::vector<float>> rows = {
        {1.0f, 10.0f, -5.0f},
        {2.0f, 5.0f,   0.0f},
        {0.5f, 12.0f, -8.0f},
    };
    auto mm = extract_min_max(rows);
    ASSERT_EQ(mm.mins.size(), 3U);
    EXPECT_FLOAT_EQ(mm.mins[0], 0.5f);
    EXPECT_FLOAT_EQ(mm.mins[1], 5.0f);
    EXPECT_FLOAT_EQ(mm.mins[2], -8.0f);
    EXPECT_FLOAT_EQ(mm.maxs[0], 2.0f);
    EXPECT_FLOAT_EQ(mm.maxs[1], 12.0f);
    EXPECT_FLOAT_EQ(mm.maxs[2], 0.0f);
}
