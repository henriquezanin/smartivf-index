#include "smartivf/utils/fvecs.hpp"
#include "smartivf/utils/ranges.hpp"
#include "smartivf/utils/scalar.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace smartivf::utils;

// RAII temporary directory. Paths are unique per test instance so that
// concurrent ctest runs do not collide.
class TempDir {
public:
    TempDir() : dir_(fs::temp_directory_path() /
                     ("smartivf-test-" + std::to_string(::getpid()) + "-" +
                      std::to_string(reinterpret_cast<std::uintptr_t>(this)))) {
        fs::create_directories(dir_);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(dir_, ec); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    [[nodiscard]] fs::path operator/(std::string_view child) const { return dir_ / child; }
private:
    fs::path dir_;
};

void write_le_i32(std::ofstream& f, std::int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_le_f32(std::ofstream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

}  // namespace

TEST(Fvecs, RoundTrip_2x3) {
    TempDir tmp;
    const auto path = tmp / "vecs.fvecs";
    {
        std::ofstream f(path, std::ios::binary);
        for (const std::vector<float> row : {std::vector<float>{1.0f, 2.0f, 3.0f},
                                             std::vector<float>{4.5f, 5.5f, 6.5f}}) {
            write_le_i32(f, 3);
            for (float v : row) write_le_f32(f, v);
        }
    }
    auto r = read_fvecs(path);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 2U);
    EXPECT_EQ((*r)[0], (std::vector<float>{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ((*r)[1], (std::vector<float>{4.5f, 5.5f, 6.5f}));
}

TEST(Fvecs, RejectsInconsistentDim) {
    TempDir tmp;
    const auto path = tmp / "bad.fvecs";
    {
        std::ofstream f(path, std::ios::binary);
        write_le_i32(f, 2); write_le_f32(f, 1); write_le_f32(f, 2);
        write_le_i32(f, 3); write_le_f32(f, 1); write_le_f32(f, 2); write_le_f32(f, 3);
    }
    auto r = read_fvecs(path);
    EXPECT_FALSE(r.has_value());
}

TEST(Ivecs, RoundTrip) {
    TempDir tmp;
    const auto path = tmp / "gt.ivecs";
    {
        std::ofstream f(path, std::ios::binary);
        std::vector<std::vector<std::int32_t>> rows = {{0, 1, 2, 3}, {7, 6, 5, 4}};
        for (const auto& row : rows) {
            write_le_i32(f, 4);
            for (std::int32_t v : row) write_le_i32(f, v);
        }
    }
    auto r = read_ivecs(path);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 2U);
    EXPECT_EQ((*r)[0], (std::vector<std::int32_t>{0, 1, 2, 3}));
    EXPECT_EQ((*r)[1], (std::vector<std::int32_t>{7, 6, 5, 4}));
}

TEST(RangesCSV, TwoDimensions) {
    TempDir tmp;
    const auto path = tmp / "ranges.csv";
    {
        std::ofstream f(path);
        f << "0.1,0.5,10.0,20.0\n";
        f << "0.3,0.9,5.0,15.0\n";
    }
    auto r = read_ranges_csv(path);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 2U);
    EXPECT_FLOAT_EQ((*r)[0][0][0], 0.1f);
    EXPECT_FLOAT_EQ((*r)[0][0][1], 0.5f);
    EXPECT_FLOAT_EQ((*r)[0][1][0], 10.0f);
    EXPECT_FLOAT_EQ((*r)[0][1][1], 20.0f);
    EXPECT_FLOAT_EQ((*r)[1][0][0], 0.3f);
    EXPECT_FLOAT_EQ((*r)[1][1][1], 15.0f);
}

TEST(RangesCSV, RejectsOddColumnCount) {
    TempDir tmp;
    const auto path = tmp / "bad.csv";
    {
        std::ofstream f(path);
        f << "0.1,0.5,10.0\n";   // 3 columns
    }
    auto r = read_ranges_csv(path);
    EXPECT_FALSE(r.has_value());
}

TEST(RangesCSV, SkipsHeaderWhenAsked) {
    TempDir tmp;
    const auto path = tmp / "h.csv";
    {
        std::ofstream f(path);
        f << "min0,max0\n";
        f << "1.0,2.0\n";
    }
    auto r = read_ranges_csv(path, true);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 1U);
    EXPECT_FLOAT_EQ((*r)[0][0][0], 1.0f);
    EXPECT_FLOAT_EQ((*r)[0][0][1], 2.0f);
}

TEST(ScalarCSV, TwoRows) {
    TempDir tmp;
    const auto path = tmp / "scalars.csv";
    {
        std::ofstream f(path);
        f << "0.312,14.7\n";
        f << "0.851,9.2\n";
    }
    auto r = read_scalar_csv(path);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->size(), 2U);
    EXPECT_FLOAT_EQ((*r)[0][0], 0.312f);
    EXPECT_FLOAT_EQ((*r)[1][1], 9.2f);
}
