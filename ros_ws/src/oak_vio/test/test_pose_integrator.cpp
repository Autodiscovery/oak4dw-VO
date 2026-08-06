#include "oak_vio/pose_integrator.hpp"

#include <gtest/gtest.h>

#include <random>

#include "synthetic_scene.hpp"

using namespace oak_vio;         // NOLINT(build/namespaces)
using namespace oak_vio::test;   // NOLINT(build/namespaces)

namespace {

/// Apply the perturbation convention used throughout:
///     T_true = Pert(delta) * T,  Pert((rho, phi)) = (Exp(phi) R, Exp(phi) t + rho)
Pose perturb(const Pose& pose, const Vec6& delta) {
    const Mat3 dR = expSO3(delta.tail<3>());
    Pose out;
    out.R = dR * pose.R;
    out.t = dR * pose.t + delta.head<3>();
    return out;
}

/// Recover the delta that maps `reference` to `perturbed`, i.e. the inverse of
/// perturb().
Vec6 difference(const Pose& perturbed, const Pose& reference) {
    Vec6 delta;
    const Mat3 dR = perturbed.R * reference.R.transpose();
    delta.tail<3>() = logSO3(dR);
    delta.head<3>() = perturbed.t - dR * reference.t;
    return delta;
}

Pose randomPose(std::mt19937_64& rng, double rotationScale = 0.4, double translationScale = 1.0) {
    std::normal_distribution<double> dist(0.0, 1.0);
    return makePose(Vec3(dist(rng), dist(rng), dist(rng)) * rotationScale, Vec3(dist(rng), dist(rng), dist(rng)) * translationScale);
}

}  // namespace

// These Jacobians were derived by hand for a non-standard (decoupled
// translation) perturbation, so they get checked against finite differences
// rather than trusted.
TEST(PoseIntegrator, InverseJacobianMatchesNumericalDerivative) {
    std::mt19937_64 rng(31337);
    constexpr double kStep = 1e-6;

    for(int trial = 0; trial < 50; ++trial) {
        const Pose pose = randomPose(rng);
        const Pose inverted = pose.inverse();
        const Mat6 analytic = PoseIntegrator::inverseJacobian(pose);

        Mat6 numerical = Mat6::Zero();
        for(int k = 0; k < 6; ++k) {
            Vec6 delta = Vec6::Zero();
            delta(k) = kStep;
            const Vec6 plus = difference(perturb(pose, delta).inverse(), inverted);
            const Vec6 minus = difference(perturb(pose, -delta).inverse(), inverted);
            numerical.col(k) = (plus - minus) / (2.0 * kStep);
        }

        const double scale = std::max(1.0, analytic.cwiseAbs().maxCoeff());
        EXPECT_LT((analytic - numerical).cwiseAbs().maxCoeff() / scale, 1e-6) << "trial " << trial;
    }
}

TEST(PoseIntegrator, ComposeJacobianMatchesNumericalDerivative) {
    std::mt19937_64 rng(4711);
    constexpr double kStep = 1e-6;

    for(int trial = 0; trial < 50; ++trial) {
        const Pose a = randomPose(rng);
        const Pose b = randomPose(rng);
        const Pose c = a * b;
        const Mat6 analytic = PoseIntegrator::composeJacobianRhs(a, b, c);

        Mat6 numerical = Mat6::Zero();
        for(int k = 0; k < 6; ++k) {
            Vec6 delta = Vec6::Zero();
            delta(k) = kStep;
            const Vec6 plus = difference(a * perturb(b, delta), c);
            const Vec6 minus = difference(a * perturb(b, -delta), c);
            numerical.col(k) = (plus - minus) / (2.0 * kStep);
        }

        const double scale = std::max(1.0, analytic.cwiseAbs().maxCoeff());
        EXPECT_LT((analytic - numerical).cwiseAbs().maxCoeff() / scale, 1e-6) << "trial " << trial;
    }
}

TEST(PoseIntegrator, LeftHandSideJacobianIsIdentity) {
    // The derivation claims d(delta_c)/d(delta_a) == I for c = a * b, which is
    // why PoseIntegrator::update simply adds the anchor covariance.
    std::mt19937_64 rng(2024);
    constexpr double kStep = 1e-6;

    const Pose a = randomPose(rng);
    const Pose b = randomPose(rng);
    const Pose c = a * b;

    Mat6 numerical = Mat6::Zero();
    for(int k = 0; k < 6; ++k) {
        Vec6 delta = Vec6::Zero();
        delta(k) = kStep;
        const Vec6 plus = difference(perturb(a, delta) * b, c);
        const Vec6 minus = difference(perturb(a, -delta) * b, c);
        numerical.col(k) = (plus - minus) / (2.0 * kStep);
    }
    EXPECT_LT((numerical - Mat6::Identity()).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(PoseIntegrator, AccumulatesPoseAcrossKeyframes) {
    PoseIntegrator integrator;
    const Mat6 covariance = Mat6::Identity() * 1e-6;

    // Three successive 10 cm forward steps, re-anchoring each time. In the
    // camera optical frame, forward is +z, and keyframe->current is the
    // inverse of the camera's own motion.
    Pose step;
    step.t = Vec3(0.0, 0.0, -0.10);

    for(int i = 0; i < 3; ++i) {
        integrator.update(step, covariance);
        integrator.anchorAtCurrent();
    }

    EXPECT_NEAR(integrator.worldFromCamera().t.z(), 0.30, 1e-9);
    EXPECT_NEAR(integrator.worldFromCamera().t.x(), 0.0, 1e-9);
}

TEST(PoseIntegrator, CovarianceGrowsMonotonicallyAcrossKeyframes) {
    PoseIntegrator integrator;
    const Mat6 covariance = Mat6::Identity() * 1e-6;
    Pose step;
    step.t = Vec3(0.0, 0.0, -0.10);

    double previousTrace = 0.0;
    for(int i = 0; i < 10; ++i) {
        integrator.update(step, covariance);
        const double trace = integrator.covariance().trace();
        EXPECT_GT(trace, previousTrace) << "drift uncertainty must accumulate at step " << i;
        previousTrace = trace;
        integrator.anchorAtCurrent();
    }
}

TEST(PoseIntegrator, CovarianceStaysSymmetricAndFinite) {
    PoseIntegrator integrator;
    std::mt19937_64 rng(5150);
    Mat6 covariance = Mat6::Identity() * 1e-5;
    covariance(0, 3) = covariance(3, 0) = 1e-7;

    for(int i = 0; i < 50; ++i) {
        integrator.update(randomPose(rng, 0.05, 0.05), covariance);
        if(i % 5 == 0) {
            integrator.anchorAtCurrent();
        }
        const Mat6 current = integrator.covariance();
        ASSERT_TRUE(current.allFinite());
        EXPECT_LT((current - current.transpose()).cwiseAbs().maxCoeff(), 1e-12);
    }
}

TEST(PoseIntegrator, ResetClearsEverything) {
    PoseIntegrator integrator;
    Pose step;
    step.t = Vec3(0.0, 0.0, -1.0);
    integrator.update(step, Mat6::Identity() * 1e-4);
    ASSERT_GT(integrator.worldFromCamera().t.norm(), 0.5);

    integrator.reset();
    EXPECT_NEAR(integrator.worldFromCamera().t.norm(), 0.0, 1e-12);
    EXPECT_NEAR(integrator.covariance().trace(), 0.0, 1e-12);
}
