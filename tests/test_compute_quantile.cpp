#include "smartivf/compute_quantile.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace smartivf;

namespace {
std::shared_ptr<Object> make_obj(std::int32_t id, std::uint64_t sfc,
                                 std::vector<float> scalars = {1.0f}) {
    auto o = std::make_shared<Object>();
    o->id = id;
    o->sfc_value = sfc;
    o->scalar_attributes = std::move(scalars);
    return o;
}
}  // namespace

TEST(ComputeQuantile, IntegerEquiDepthSplit_10Objects_4Parts) {
    // Expected integer-arithmetic boundaries for n=10, p=4:
    //   [0] → [0, 2)  [1] → [2, 5)  [2] → [5, 7)  [3] → [7, 10)
    std::vector<std::shared_ptr<Object>> objs;
    for (std::int32_t i = 0; i < 10; ++i) {
        objs.push_back(make_obj(i, static_cast<std::uint64_t>(i * 10)));
    }
    auto r = compute_quantile(objs, 4);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->objects_per_partition.size(), 4U);
    EXPECT_EQ(r->objects_per_partition[0].size(), 2U);  // [0, 2)
    EXPECT_EQ(r->objects_per_partition[1].size(), 3U);  // [2, 5)
    EXPECT_EQ(r->objects_per_partition[2].size(), 2U);  // [5, 7)
    EXPECT_EQ(r->objects_per_partition[3].size(), 3U);  // [7, 10)
    EXPECT_EQ(r->partitions[0].min_sfc, 0U);
    EXPECT_EQ(r->partitions[3].max_sfc, 90U);
}

TEST(ComputeQuantile, EmptyInput) {
    std::vector<std::shared_ptr<Object>> empty;
    auto r = compute_quantile(empty, 4);
    EXPECT_FALSE(r.has_value());
}

TEST(ComputeQuantile, MorePartitionsThanObjects) {
    std::vector<std::shared_ptr<Object>> objs;
    objs.push_back(make_obj(0, 0));
    objs.push_back(make_obj(1, 1));
    auto r = compute_quantile(objs, 4);
    EXPECT_FALSE(r.has_value());
}

TEST(ComputeQuantile, ScalarBoundsAreComputedPerPartition) {
    std::vector<std::shared_ptr<Object>> objs;
    objs.push_back(make_obj(0, 0,  {1.0f, 100.0f}));
    objs.push_back(make_obj(1, 10, {2.0f, 50.0f}));
    objs.push_back(make_obj(2, 20, {3.0f, 25.0f}));
    objs.push_back(make_obj(3, 30, {4.0f, 5.0f}));
    auto r = compute_quantile(objs, 2);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ASSERT_EQ(r->partitions.size(), 2U);
    // First partition: rows 0-1 → mins {1, 50}, maxs {2, 100}
    EXPECT_FLOAT_EQ(r->partitions[0].min_scalar_partition_attributes[0], 1.0f);
    EXPECT_FLOAT_EQ(r->partitions[0].max_scalar_partition_attributes[1], 100.0f);
    // Second partition: rows 2-3 → mins {3, 5}, maxs {4, 25}
    EXPECT_FLOAT_EQ(r->partitions[1].min_scalar_partition_attributes[0], 3.0f);
    EXPECT_FLOAT_EQ(r->partitions[1].max_scalar_partition_attributes[1], 25.0f);
}
