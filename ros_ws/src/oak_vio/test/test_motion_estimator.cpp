#include "oak_vio/motion_estimator.hpp"

#include <gtest/gtest.h>

#include "synthetic_scene.hpp"

using namespace oak_vio;         // NOLINT(build/namespaces)
using namespace oak_vio::test;   // NOLINT(build/namespaces)

namespace {

MotionEstimator makeEstimator(VioParams params = VioParams{}) {
    return {makeTestCamera(), params};
}

std::vector<int> allIndices(std::size_t count) {
    std::vector<int> indices(count);
    for(std::size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<int>(i);
    }
    return indices;
}

}  // namespace

// The single most important test in the suite: if the analytic Jacobian is
// wrong, Gauss-Newton still converges to *something* and every downstream
// number is quietly biased.
TEST(MotionEstimator, AnalyticJacobianMatchesNumericalDerivative) {
    const MotionEstimator estimator = makeEstimator();
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<double> xy(-4.0, 4.0);
    std::uniform_real_distribution<double> depth(1.0, 20.0);
    std::normal_distribution<double> small(0.0, 0.3);

    constexpr double kStep = 1e-7;
    double worstRelativeError = 0.0;
    int tested = 0;

    while(tested < 200) {
        Correspondence c;
        c.pointKf = Vec3(xy(rng), xy(rng), depth(rng));
        c.depthKf = c.pointKf.z();
        c.disparityKf = estimator.camera().fx() * estimator.camera().baseline() / c.pointKf.z();

        const Pose pose = makePose(Vec3(small(rng), small(rng), small(rng)), Vec3(small(rng), small(rng), small(rng)));
        if(pose.apply(c.pointKf).z() <= 0.5) {
            continue;
        }
        ++tested;
        c.zCur = estimator.camera().project(pose.apply(c.pointKf)) + Vec3(0.3, -0.2, 0.1);

        Vec3 residual;
        Mat36 analytic;
        ASSERT_TRUE(estimator.residualAndJacobian(c, pose, residual, analytic));

        Mat36 numerical = Mat36::Zero();
        for(int k = 0; k < 6; ++k) {
            Vec6 delta = Vec6::Zero();
            delta(k) = kStep;

            const auto perturb = [&](const Vec6& d) {
                Pose out;
                const Mat3 dR = expSO3(d.tail<3>());
                out.R = dR * pose.R;
                out.t = dR * pose.t + d.head<3>();
                return out;
            };

            const Vec3 plus = estimator.camera().project(perturb(delta).apply(c.pointKf)) - c.zCur;
            const Vec3 minus = estimator.camera().project(perturb(-delta).apply(c.pointKf)) - c.zCur;
            numerical.col(k) = (plus - minus) / (2.0 * kStep);
        }

        const double scale = std::max(1.0, analytic.cwiseAbs().maxCoeff());
        worstRelativeError = std::max(worstRelativeError, (analytic - numerical).cwiseAbs().maxCoeff() / scale);
    }

    EXPECT_LT(worstRelativeError, 1e-6) << "worst relative error " << worstRelativeError;
}

TEST(MotionEstimator, RecoversPoseExactlyWithoutNoise) {
    const MotionEstimator estimator = makeEstimator();
    SyntheticScene scene(makeTestCamera(), 1280, 800, 7);

    for(int trial = 0; trial < 20; ++trial) {
        scene.generatePoints(200, 1.0, 25.0);
        const Pose truth = makePose(Vec3(0.02 * trial - 0.15, 0.03, -0.01), Vec3(0.1, -0.05, 0.25));

        const auto keyframe = scene.observe(Pose{}, 0.0, 0.0);
        const auto current = scene.observe(truth, 0.0, 0.0);
        const auto correspondences = scene.correspondences(keyframe, current);
        ASSERT_GE(correspondences.size(), 30U);

        // The estimator solves for keyframe->current, which is the inverse of
        // the camera's own motion.
        const Pose expected = truth.inverse();

        Pose seed;
        ASSERT_TRUE(estimator.solveClosedForm(correspondences, allIndices(correspondences.size()), seed));
        const RefinementResult refined = estimator.refine(correspondences, allIndices(correspondences.size()), seed);

        ASSERT_FALSE(refined.degenerate);
        EXPECT_LT(rotationError(expected, refined.pose), 1e-8);
        EXPECT_LT((expected.t - refined.pose.t).norm(), 1e-8);
    }
}

TEST(MotionEstimator, ClosedFormIsExactFromThreePoints) {
    const MotionEstimator estimator = makeEstimator();
    SyntheticScene scene(makeTestCamera(), 1280, 800, 11);
    scene.generatePoints(80, 1.0, 20.0);

    const Pose truth = makePose(Vec3(0.03, -0.02, 0.01), Vec3(0.15, 0.0, 0.3));
    const auto correspondences = scene.correspondences(scene.observe(Pose{}, 0.0, 0.0), scene.observe(truth, 0.0, 0.0));
    ASSERT_GE(correspondences.size(), 20U);
    const Pose expected = truth.inverse();

    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::size_t> pick(0, correspondences.size() - 1);
    for(int trial = 0; trial < 50; ++trial) {
        std::array<int, 3> sample{};
        sample[0] = static_cast<int>(pick(rng));
        do {
            sample[1] = static_cast<int>(pick(rng));
        } while(sample[1] == sample[0]);
        do {
            sample[2] = static_cast<int>(pick(rng));
        } while(sample[2] == sample[0] || sample[2] == sample[1]);

        Pose solved;
        ASSERT_TRUE(estimator.solveMinimal(correspondences, sample, solved));
        EXPECT_LT(rotationError(expected, solved), 1e-8);
        EXPECT_LT((expected.t - solved.t).norm(), 1e-8);
    }
}

TEST(MotionEstimator, RejectsCollinearMinimalSet) {
    const MotionEstimator estimator = makeEstimator();
    const RectifiedCamera camera = makeTestCamera();

    // Three points along a line: rotation about that line is unconstrained.
    std::vector<Correspondence> correspondences;
    const Vec3 base(0.5, 0.2, 6.0);
    const Vec3 direction(1.0, 0.3, 0.2);
    for(double s : {-1.0, 0.0, 1.0}) {
        Correspondence c;
        c.pointKf = base + s * direction;
        c.depthKf = c.pointKf.z();
        c.disparityKf = camera.fx() * camera.baseline() / c.pointKf.z();
        c.zCur = camera.project(c.pointKf);
        correspondences.push_back(c);
    }

    Pose solved;
    EXPECT_FALSE(estimator.solveMinimal(correspondences, {0, 1, 2}, solved));
}

// Regression guard for the error-in-variables bug. Treating the triangulated
// keyframe point as exact is fine on near scenes and produces occasional
// metre-scale divergence on far ones.
TEST(MotionEstimator, DepthUncertaintyWeightingPreventsFarSceneDivergence) {
    SyntheticScene scene(makeTestCamera(), 1280, 800, 20260806);
    const MotionEstimator estimator = makeEstimator();

    std::vector<double> weightedErrors;
    std::vector<double> naiveErrors;

    for(int trial = 0; trial < 40; ++trial) {
        scene.generatePoints(300, 5.0, 40.0);
        const Pose truth = makePose(Vec3(0.01, 0.02, -0.01), Vec3(0.05, 0.02, 0.10));

        // Noisy disparity at the keyframe is what makes the keyframe point
        // uncertain in the first place.
        const auto keyframe = scene.observe(Pose{}, 0.3, 0.5);
        const auto current = scene.observe(truth, 0.3, 0.5);
        auto correspondences = scene.correspondences(keyframe, current);
        if(correspondences.size() < 50) {
            continue;
        }
        const Pose expected = truth.inverse();
        const auto indices = allIndices(correspondences.size());

        Pose seed;
        if(!estimator.solveClosedForm(correspondences, indices, seed)) {
            continue;
        }
        weightedErrors.push_back((expected.t - estimator.refine(correspondences, indices, seed).pose.t).norm());

        // Zeroing disparityKf disables the propagation, reproducing the naive
        // behaviour through the same code path.
        auto naive = correspondences;
        for(auto& c : naive) {
            c.disparityKf = 0.0;
        }
        naiveErrors.push_back((expected.t - estimator.refine(naive, indices, seed).pose.t).norm());
    }

    ASSERT_GE(weightedErrors.size(), 20U);
    std::sort(weightedErrors.begin(), weightedErrors.end());
    std::sort(naiveErrors.begin(), naiveErrors.end());

    const auto percentile = [](const std::vector<double>& v, double p) {
        return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))];
    };

    // The tail is the point: the median difference is modest, the worst case
    // is not.
    EXPECT_LT(percentile(weightedErrors, 0.95), 0.05) << "weighted p95 should stay in the centimetres";
    EXPECT_LT(percentile(weightedErrors, 0.95), percentile(naiveErrors, 0.95));
}

TEST(MotionEstimator, CovarianceIsFiniteAndInflatedWhenRankDeficient) {
    const MotionEstimator estimator = makeEstimator();

    const Mat6 covariance = estimator.covarianceFromInformation(Mat6::Zero(), 50);
    ASSERT_TRUE(covariance.allFinite());
    EXPECT_GT(covariance(0, 0), 1e3) << "an uninformative solve must report large uncertainty";

    // A well-conditioned solve should produce a small, symmetric covariance.
    const Mat6 wellConditioned = Mat6::Identity() * 1e6;
    const Mat6 good = estimator.covarianceFromInformation(wellConditioned, 200);
    EXPECT_TRUE(good.allFinite());
    EXPECT_LT(good(0, 0), 1e-3);
    EXPECT_NEAR((good - good.transpose()).cwiseAbs().maxCoeff(), 0.0, 1e-12);
}

TEST(MotionEstimator, CovarianceGrowsAsInliersFall) {
    const MotionEstimator estimator = makeEstimator();
    const Mat6 information = Mat6::Identity() * 1e5;
    const Mat6 many = estimator.covarianceFromInformation(information, 400);
    const Mat6 few = estimator.covarianceFromInformation(information, 15);
    EXPECT_GT(few(0, 0), many(0, 0));
}
