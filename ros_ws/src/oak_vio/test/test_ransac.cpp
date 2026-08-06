#include "oak_vio/ransac.hpp"

#include <gtest/gtest.h>

#include "synthetic_scene.hpp"

using namespace oak_vio;         // NOLINT(build/namespaces)
using namespace oak_vio::test;   // NOLINT(build/namespaces)

namespace {

RansacMotionSolver makeSolver(VioParams params = VioParams{}) {
    return {MotionEstimator(makeTestCamera(), params), params, 12345};
}

}  // namespace

TEST(Ransac, SolvesCleanDataAndMarksEverythingInlier) {
    auto solver = makeSolver();
    SyntheticScene scene(makeTestCamera(), 1280, 800, 3);
    scene.generatePoints(250, 1.0, 20.0);

    const Pose truth = makePose(Vec3(0.02, -0.01, 0.005), Vec3(0.08, 0.0, 0.2));
    const auto correspondences = scene.correspondences(scene.observe(Pose{}, 0.0, 0.0), scene.observe(truth, 0.0, 0.0));
    ASSERT_GE(correspondences.size(), 50U);

    const RansacResult result = solver.solve(correspondences, Pose{});
    ASSERT_TRUE(result.success) << result.rejectionReason;
    EXPECT_GT(result.inlierRatio, 0.99);
    EXPECT_LT(rotationError(truth.inverse(), result.pose), 1e-6);
    EXPECT_LT((truth.inverse().t - result.pose.t).norm(), 1e-6);
}

TEST(Ransac, SurvivesHeavyOutlierContamination) {
    auto solver = makeSolver();
    SyntheticScene scene(makeTestCamera(), 1280, 800, 5);
    scene.generatePoints(400, 1.0, 20.0);

    const Pose truth = makePose(Vec3(0.01, 0.03, -0.02), Vec3(0.05, -0.02, 0.15));
    const auto keyframe = scene.observe(Pose{}, 0.3, 0.5);
    const auto current = scene.observe(truth, 0.3, 0.5, /*outlierFraction=*/0.30);
    const auto correspondences = scene.correspondences(keyframe, current);
    ASSERT_GE(correspondences.size(), 100U);

    const RansacResult result = solver.solve(correspondences, Pose{});
    ASSERT_TRUE(result.success) << result.rejectionReason;

    // With 30% gross outliers the inlier ratio should land near 0.7, and the
    // estimate should stay accurate to a few millimetres.
    EXPECT_GT(result.inlierRatio, 0.5);
    EXPECT_LT(result.inlierRatio, 0.95);
    EXPECT_LT(rotationError(truth.inverse(), result.pose), 2e-3);
    EXPECT_LT((truth.inverse().t - result.pose.t).norm(), 0.02);
}

TEST(Ransac, RejectsImplausibleMotion) {
    VioParams params;
    params.maxTranslationPerSpanM = 0.01;  // 1 cm, far below the real motion
    auto solver = makeSolver(params);

    SyntheticScene scene(makeTestCamera(), 1280, 800, 9);
    scene.generatePoints(200, 1.0, 20.0);
    const Pose truth = makePose(Vec3::Zero(), Vec3(0.0, 0.0, 1.0));
    const auto correspondences = scene.correspondences(scene.observe(Pose{}, 0.0, 0.0), scene.observe(truth, 0.0, 0.0));
    ASSERT_GE(correspondences.size(), 30U);

    const RansacResult result = solver.solve(correspondences, Pose{});
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.rejected);
}

TEST(Ransac, FailsCleanlyOnTooFewCorrespondences) {
    auto solver = makeSolver();
    const std::vector<Correspondence> two(2);
    const RansacResult result = solver.solve(two, Pose{});
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.rejected);
    EXPECT_EQ(result.numCorrespondences, 2);
}

TEST(Ransac, GoodSeedReducesIterations) {
    // The prior is evaluated as a hypothesis in its own right, so a correct
    // seed should collapse the adaptive iteration budget.
    SyntheticScene scene(makeTestCamera(), 1280, 800, 17);
    scene.generatePoints(300, 1.0, 20.0);
    const Pose truth = makePose(Vec3(0.01, 0.0, 0.0), Vec3(0.03, 0.0, 0.12));
    const auto correspondences = scene.correspondences(scene.observe(Pose{}, 0.2, 0.3), scene.observe(truth, 0.2, 0.3));
    ASSERT_GE(correspondences.size(), 80U);

    auto coldSolver = makeSolver();
    auto warmSolver = makeSolver();
    const RansacResult cold = coldSolver.solve(correspondences, Pose{});
    const RansacResult warm = warmSolver.solve(correspondences, truth.inverse());

    ASSERT_TRUE(cold.success);
    ASSERT_TRUE(warm.success);
    EXPECT_LE(warm.iterationsUsed, cold.iterationsUsed);
    EXPECT_LT((warm.pose.t - cold.pose.t).norm(), 0.01) << "seed must not change the answer, only the cost";
}

TEST(Ransac, IsDeterministicForAGivenSeed) {
    SyntheticScene scene(makeTestCamera(), 1280, 800, 23);
    scene.generatePoints(200, 1.0, 20.0);
    const Pose truth = makePose(Vec3(0.02, 0.01, 0.0), Vec3(0.05, 0.0, 0.1));
    const auto correspondences = scene.correspondences(scene.observe(Pose{}, 0.3, 0.5), scene.observe(truth, 0.3, 0.5, 0.15));

    auto first = makeSolver();
    auto second = makeSolver();
    const RansacResult a = first.solve(correspondences, Pose{});
    const RansacResult b = second.solve(correspondences, Pose{});

    ASSERT_TRUE(a.success);
    ASSERT_TRUE(b.success);
    EXPECT_EQ(a.inliers.size(), b.inliers.size());
    EXPECT_NEAR((a.pose.t - b.pose.t).norm(), 0.0, 1e-15);
}
