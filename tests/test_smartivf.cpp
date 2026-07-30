// End-to-end tests on deterministic synthetic datasets: build an index, run a
// filtered query whose answer is known by construction, and check that the
// expected object is returned.

#include "smartivf/smartivf.hpp"
#include "smartivf/utils/fvecs.hpp"
#include "smartivf/utils/ranges.hpp"
#include "smartivf/utils/scalar.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace smartivf;

namespace {

class TempDir {
public:
    TempDir() : dir_(fs::temp_directory_path() /
                     ("smartivf-e2e-" + std::to_string(::getpid()) + "-" +
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
void write_fvecs(const fs::path& p, const std::vector<std::vector<float>>& rows) {
    std::ofstream f(p, std::ios::binary);
    for (const auto& r : rows) {
        write_le_i32(f, static_cast<std::int32_t>(r.size()));
        for (float v : r) write_le_f32(f, v);
    }
}
void write_csv(const fs::path& p, const std::vector<std::vector<float>>& rows) {
    std::ofstream f(p);
    for (const auto& r : rows) {
        for (std::size_t i = 0; i < r.size(); ++i) {
            f << r[i];
            if (i + 1 < r.size()) f << ",";
        }
        f << "\n";
    }
}

}  // namespace

TEST(SmartIVF, EndToEnd_L2_HitsHighRecall_OnFilteredQuery) {
    TempDir tmp;
    constexpr int N = 1000;
    constexpr int Dvec = 16;
    constexpr int Dscalar = 2;

    std::mt19937_64 rng(0xABCDEF01);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> scalar_dist(0.0f, 100.0f);

    std::vector<std::vector<float>> base(N), scalars(N);
    for (int i = 0; i < N; ++i) {
        base[i].resize(Dvec);
        for (auto& v : base[i]) v = noise(rng);
        scalars[i] = {scalar_dist(rng), scalar_dist(rng)};
    }

    const auto fvecs_path  = tmp / "base.fvecs";
    const auto scalar_path = tmp / "scalars.csv";
    const auto query_path  = tmp / "queries.fvecs";
    const auto range_path  = tmp / "ranges.csv";

    write_fvecs(fvecs_path, base);
    write_csv(scalar_path, scalars);

    // Query 1: the very first base vector (perturbed); filter that admits all base.
    std::vector<std::vector<float>> queries = {base[0]};
    for (auto& v : queries[0]) v += 0.01f * noise(rng);
    std::vector<std::vector<float>> ranges = {{0.0f, 100.0f, 0.0f, 100.0f}};

    write_fvecs(query_path, queries);
    write_csv(range_path, ranges);

    SmartIVFConfig cfg{
        .scalar_dimensions          = Dscalar,
        .scalar_bits_per_dimension  = 8,
        .partition_threshold        = 200,
        .partitions                 = 8,
        .metric                     = MetricType::L2,
        .seed                       = 42,
        .threads                    = 1,
    };
    SmartIVF s(cfg);

    auto idx = s.build_index(fvecs_path, scalar_path);
    ASSERT_TRUE(idx.has_value()) << idx.error().message;
    EXPECT_EQ(idx->partitions.size(), 8U);

    std::array<float, 2> r0 = {ranges[0][0], ranges[0][1]};
    std::array<float, 2> r1 = {ranges[0][2], ranges[0][3]};
    std::array<std::array<float, 2>, 2> qranges = {r0, r1};

    auto res = s.search(*idx, queries[0], qranges, /*k=*/10,
                        /*budget=*/256, /*minProbe=*/1, /*maxProbe=*/64,
                        /*candidateTarget=*/200);
    ASSERT_TRUE(res.has_value()) << res.error().message;
    ASSERT_FALSE(res->results.empty());

    // The query is a perturbation of base[0], so id=0 should be in the top-10.
    std::unordered_set<std::int32_t> top10;
    for (const auto& o : res->results) top10.insert(o->id);
    EXPECT_TRUE(top10.contains(0));
}

TEST(SmartIVF, EndToEnd_Cosine_BasicSanity) {
    TempDir tmp;
    constexpr int N = 500;
    constexpr int Dvec = 8;

    std::mt19937_64 rng(0x12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> scalar_dist(0.0f, 100.0f);

    std::vector<std::vector<float>> base(N), scalars(N);
    for (int i = 0; i < N; ++i) {
        base[i].resize(Dvec);
        for (auto& v : base[i]) v = nd(rng);
        scalars[i] = {scalar_dist(rng)};
    }

    write_fvecs(tmp / "base.fvecs", base);
    write_csv(tmp / "scalars.csv", scalars);
    write_fvecs(tmp / "q.fvecs", {base[5]});
    write_csv(tmp / "r.csv", {{0.0f, 100.0f}});

    SmartIVFConfig cfg{
        .scalar_dimensions          = 1,
        .scalar_bits_per_dimension  = 8,
        .partition_threshold        = 100,
        .partitions                 = 4,
        .metric                     = MetricType::Cosine,
        .seed                       = 42,
        .threads                    = 1,
    };
    SmartIVF s(cfg);
    auto idx = s.build_index(tmp / "base.fvecs", tmp / "scalars.csv");
    ASSERT_TRUE(idx.has_value()) << idx.error().message;

    std::array<std::array<float, 2>, 1> qr = {{ {0.0f, 100.0f} }};
    auto res = s.search(*idx, base[5], qr, 5, 256, 1, 32, 100);
    ASSERT_TRUE(res.has_value()) << res.error().message;
    ASSERT_FALSE(res->results.empty());

    std::unordered_set<std::int32_t> top5;
    for (const auto& o : res->results) top5.insert(o->id);
    EXPECT_TRUE(top5.contains(5));
}
