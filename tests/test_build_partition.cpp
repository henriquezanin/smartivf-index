#include "smartivf/build_partition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

using namespace smartivf;

namespace {
std::shared_ptr<Object> make_obj(std::int32_t id, std::uint64_t sfc,
                                 std::vector<float> emb) {
    auto o = std::make_shared<Object>();
    o->id = id;
    o->sfc_value = sfc;
    o->embedding = std::move(emb);
    o->scalar_attributes = {1.0f};
    return o;
}
}  // namespace

TEST(BuildPartition, Flat_SortsBySFCValue) {
    std::vector<std::shared_ptr<Object>> objs;
    objs.push_back(make_obj(0, 30, {1.0f, 2.0f}));
    objs.push_back(make_obj(1, 10, {3.0f, 4.0f}));
    objs.push_back(make_obj(2, 20, {5.0f, 6.0f}));
    Partition p;
    build_flat_partition(p, objs);
    EXPECT_EQ(p.type, PartitionType::Flat);
    ASSERT_EQ(p.objects.size(), 3U);
    EXPECT_EQ(p.objects[0]->sfc_value, 10U);
    EXPECT_EQ(p.objects[1]->sfc_value, 20U);
    EXPECT_EQ(p.objects[2]->sfc_value, 30U);
}

TEST(BuildPartition, Kmeans_BuildsCentroidsAndInvertedLists) {
    // 100 random 8-D vectors clustered loosely into 2 modes → floor(sqrt(100))=10 clusters.
    std::mt19937_64 rng(0xCAFEBABE);
    std::normal_distribution<float> n1(0.0f, 0.1f);
    std::normal_distribution<float> n2(5.0f, 0.1f);
    std::vector<std::shared_ptr<Object>> objs;
    objs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        std::vector<float> emb(8);
        auto& which = (i % 2 == 0) ? n1 : n2;
        for (float& v : emb) v = which(rng);
        objs.push_back(make_obj(static_cast<std::int32_t>(i),
                                static_cast<std::uint64_t>(i), std::move(emb)));
    }

    Partition p;
    auto r = build_kmeans_partition(p, objs, MetricType::L2, /*seed=*/42);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(p.type, PartitionType::KMeans);
    // floor(sqrt(100)) = 10 clusters
    EXPECT_EQ(p.centroids.size(), 10U);
    EXPECT_EQ(p.centroid_norm_sq.size(), 10U);
    // Every object should land in some inverted list.
    std::size_t total_assigned = 0;
    for (const auto& [cid, list] : p.inverted_lists) total_assigned += list.size();
    EXPECT_EQ(total_assigned, 100U);
    // Each inverted list is sorted by sfc_value.
    for (const auto& [cid, list] : p.inverted_lists) {
        for (std::size_t i = 1; i < list.size(); ++i) {
            EXPECT_LE(list[i - 1]->sfc_value, list[i]->sfc_value);
        }
    }
}

TEST(BuildPartition, Kmeans_RespectsClusterCountClamp) {
    // n=4 → floor(sqrt(4))=2 clusters
    std::vector<std::shared_ptr<Object>> objs;
    for (int i = 0; i < 4; ++i) {
        objs.push_back(make_obj(i, static_cast<std::uint64_t>(i),
                                {static_cast<float>(i), 0.0f}));
    }
    Partition p;
    auto r = build_kmeans_partition(p, objs, MetricType::L2, 42);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(p.centroids.size(), 2U);
}

TEST(BuildPartition, Kmeans_CosineNormalisesCentroids) {
    std::mt19937_64 rng(0xDEADBEEF);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::shared_ptr<Object>> objs;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> emb(4);
        float n = 0.0f;
        for (float& v : emb) { v = dist(rng); n += v * v; }
        n = std::sqrt(n);
        for (float& v : emb) v /= n;  // L2-normalise (the caller's responsibility for cosine)
        objs.push_back(make_obj(i, static_cast<std::uint64_t>(i), std::move(emb)));
    }
    Partition p;
    auto r = build_kmeans_partition(p, objs, MetricType::Cosine, 42);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    for (const auto& c : p.centroids) {
        float n2 = 0.0f;
        for (float v : c) n2 += v * v;
        // Centroids should be on the unit sphere (within fp tolerance).
        EXPECT_NEAR(n2, 1.0f, 1e-4f);
    }
}
