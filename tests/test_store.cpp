// Round-trip store / load. Builds a small index, serialises it, reads it back,
// and compares the partition metadata of the two indices.

#include "smartivf/smartivf.hpp"
#include "smartivf/store.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using namespace smartivf;

namespace {

class TempDir {
public:
    TempDir() : dir_(fs::temp_directory_path() /
                     ("smartivf-store-" + std::to_string(::getpid()) + "-" +
                      std::to_string(reinterpret_cast<std::uintptr_t>(this)))) {
        fs::create_directories(dir_);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(dir_, ec); }
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

TEST(Store, RoundTrip_PreservesIndexStructure) {
    TempDir tmp;
    constexpr int N = 300;
    constexpr int Dvec = 8;

    std::mt19937_64 rng(0xBEEFCAFE);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> sd(0.0f, 100.0f);

    {
        std::ofstream f(tmp / "base.fvecs", std::ios::binary);
        for (int i = 0; i < N; ++i) {
            write_le_i32(f, Dvec);
            for (int d = 0; d < Dvec; ++d) write_le_f32(f, nd(rng));
        }
    }
    {
        std::ofstream f(tmp / "scalars.csv");
        for (int i = 0; i < N; ++i) f << sd(rng) << "," << sd(rng) << "\n";
    }

    SmartIVFConfig cfg{
        .scalar_dimensions          = 2,
        .scalar_bits_per_dimension  = 8,
        .partition_threshold        = 100,
        .partitions                 = 4,
        .metric                     = MetricType::L2,
        .seed                       = 42,
        .threads                    = 1,
    };
    SmartIVF s(cfg);
    auto idx = s.build_index(tmp / "base.fvecs", tmp / "scalars.csv");
    ASSERT_TRUE(idx.has_value()) << idx.error().message;

    const auto idx_path = tmp / "index.bin";
    auto saved = store_index(*idx, idx_path);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    auto loaded = load_index(idx_path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

    EXPECT_EQ(loaded->partitions.size(), idx->partitions.size());
    EXPECT_EQ(loaded->min_scalar_attributes, idx->min_scalar_attributes);
    EXPECT_EQ(loaded->max_scalar_attributes, idx->max_scalar_attributes);

    // Spot-check each partition's metadata.
    for (std::size_t p = 0; p < idx->partitions.size(); ++p) {
        EXPECT_EQ(loaded->partitions[p].type,    idx->partitions[p].type);
        EXPECT_EQ(loaded->partitions[p].min_sfc, idx->partitions[p].min_sfc);
        EXPECT_EQ(loaded->partitions[p].max_sfc, idx->partitions[p].max_sfc);
    }
}

TEST(Store, RejectsBadMagic) {
    TempDir tmp;
    const auto p = tmp / "bogus.bin";
    {
        std::ofstream f(p, std::ios::binary);
        const std::uint32_t bad = 0xDEADBEEFu;
        f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    }
    auto r = load_index(p);
    EXPECT_FALSE(r.has_value());
}
