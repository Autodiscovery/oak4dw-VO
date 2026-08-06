#include "oak_vio/bucketing.hpp"

#include <gtest/gtest.h>

#include <set>

#include "synthetic_scene.hpp"

using namespace oak_vio;         // NOLINT(build/namespaces)
using namespace oak_vio::test;   // NOLINT(build/namespaces)

namespace {

Correspondence at(double u, double v, std::uint32_t age = 1, float trackingError = 1.0F) {
    Correspondence c;
    c.zCur = Vec3(u, v, 10.0);
    c.uvKf = Vec2(u, v);
    c.age = age;
    c.trackingError = trackingError;
    return c;
}

}  // namespace

TEST(Bucketing, CapsFeaturesPerCell) {
    VioParams params;
    params.bucketCols = 4;
    params.bucketRows = 2;
    params.maxFeaturesPerBucket = 3;

    // 50 features crammed into the top-left cell.
    std::vector<Correspondence> correspondences;
    for(int i = 0; i < 50; ++i) {
        correspondences.push_back(at(10.0 + i * 0.5, 10.0 + i * 0.2));
    }

    const auto selected = selectBucketed(correspondences, 1280, 800, params);
    EXPECT_EQ(selected.size(), 3U) << "one occupied cell should yield at most maxFeaturesPerBucket";
}

TEST(Bucketing, SpreadsAcrossTheImage) {
    VioParams params;
    params.bucketCols = 8;
    params.bucketRows = 5;
    params.maxFeaturesPerBucket = 2;

    // A dense clump on the left plus a sparse scatter elsewhere: without
    // bucketing the clump would dominate the solve.
    std::vector<Correspondence> correspondences;
    for(int i = 0; i < 200; ++i) {
        correspondences.push_back(at(20.0 + (i % 20), 20.0 + (i / 20)));
    }
    for(int col = 0; col < 8; ++col) {
        for(int row = 0; row < 5; ++row) {
            correspondences.push_back(at(col * 160.0 + 80.0, row * 160.0 + 80.0));
        }
    }

    const auto selected = selectBucketed(correspondences, 1280, 800, params);

    std::set<std::pair<int, int>> occupiedCells;
    for(const int index : selected) {
        const Correspondence& c = correspondences[static_cast<std::size_t>(index)];
        occupiedCells.emplace(static_cast<int>(c.zCur.x() / 160.0), static_cast<int>(c.zCur.y() / 160.0));
    }
    EXPECT_GE(occupiedCells.size(), 30U) << "selection should touch most of the 8x5 grid";
    EXPECT_LE(selected.size(), 8U * 5U * 2U);
}

TEST(Bucketing, PrefersOlderTracksWithinACell) {
    VioParams params;
    params.bucketCols = 1;
    params.bucketRows = 1;
    params.maxFeaturesPerBucket = 2;

    std::vector<Correspondence> correspondences{
        at(100.0, 100.0, /*age=*/1),
        at(200.0, 200.0, /*age=*/50),
        at(300.0, 300.0, /*age=*/25),
        at(400.0, 400.0, /*age=*/2),
    };

    const auto selected = selectBucketed(correspondences, 1280, 800, params);
    ASSERT_EQ(selected.size(), 2U);
    std::set<std::uint32_t> ages;
    for(const int index : selected) {
        ages.insert(correspondences[static_cast<std::size_t>(index)].age);
    }
    EXPECT_EQ(ages, (std::set<std::uint32_t>{25, 50}));
}

TEST(Bucketing, HandlesEmptyInput) {
    const VioParams params;
    EXPECT_TRUE(selectBucketed({}, 1280, 800, params).empty());
    EXPECT_DOUBLE_EQ(medianParallax({}, {}), 0.0);
}

TEST(Bucketing, MedianParallaxIsCorrect) {
    std::vector<Correspondence> correspondences;
    for(int i = 1; i <= 5; ++i) {
        Correspondence c;
        c.uvKf = Vec2(0.0, 0.0);
        c.zCur = Vec3(static_cast<double>(i), 0.0, 10.0);  // parallax == i
        correspondences.push_back(c);
    }
    const std::vector<int> indices{0, 1, 2, 3, 4};
    EXPECT_DOUBLE_EQ(medianParallax(correspondences, indices), 3.0);
}
