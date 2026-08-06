// All estimator tunables in one place. Every field is mirrored by a ROS
// parameter under the `vio.` namespace in stereo_vio_node.cpp, so anything
// added here should be declared there too.
#pragma once

#include <cstdint>

namespace oak_vio {

struct VioParams {
    // ---- Correspondence gating -------------------------------------------
    /// Disparity floor. Points below this are effectively at infinity: they
    /// constrain rotation well but translation not at all, and their depth
    /// error explodes. With f≈537 px and b=75 mm, 1.5 px ≈ 27 m.
    float minDisparityPx{1.5F};
    /// Disparity ceiling. Anything closer than ~f*b/max is either a bad match
    /// or too close to be stably tracked.
    float maxDisparityPx{200.0F};
    /// Reject features the HW tracker is unsure about.
    float maxTrackingError{30.0F};
    /// Fraction of width/height masked at each image edge. On the wide lens
    /// the rectified periphery is stretched enough to degrade corner quality;
    /// Phase 0 measurement sets this.
    double maskBorderFraction{0.0};

    // ---- Spatial bucketing -----------------------------------------------
    int bucketCols{8};
    int bucketRows{5};
    int maxFeaturesPerBucket{10};

    // ---- RANSAC -----------------------------------------------------------
    int ransacMinIterations{20};
    int ransacMaxIterations{200};
    /// Inlier test is on reprojection error in the current frame, in pixels.
    double ransacInlierThresholdPx{2.0};
    double ransacConfidence{0.999};
    /// Below these the solution is reported but flagged LowInliers.
    int minInliers{12};
    double minInlierRatio{0.35};

    // ---- Gauss-Newton refinement -----------------------------------------
    int gnMaxIterations{10};
    /// Convergence when the parameter update norm drops below this.
    double gnConvergenceEps{1e-7};
    /// Huber transition point, in pixels.
    double huberDeltaPx{1.5};

    // ---- Keyframe policy --------------------------------------------------
    /// False falls back to frame-to-frame tracking, for A/B drift comparison.
    bool keyframeReferenced{true};
    /// Promote a new keyframe when surviving tracks drop below this.
    int keyframeMinTracked{60};

    /// ...or when median parallax from the keyframe exceeds this, in pixels.
    ///
    /// This is the single most consequential keyframe parameter, and it is a
    /// genuine optimum rather than "bigger is better". Two effects pull
    /// against each other:
    ///
    ///   * Too small (frame-to-frame): parallax approaches zero and
    ///     translation becomes near-unobservable.
    ///   * Too large: the keyframe's own triangulation error is a fixed bias
    ///     that persists for the whole span instead of averaging out, and it
    ///     grows with span length. ORB-SLAM escapes this by refining map
    ///     points in local bundle adjustment; we do not run one.
    ///
    /// Simulation over a closed loop at four speeds (8 seeds each) found a
    /// clean crossover: frame-to-frame beat a 30-frame keyframe span by
    /// 2.2-3.6x at 6-31 mm of camera motion per frame, while a 30-frame span
    /// beat frame-to-frame by 1.5x at 3 mm/frame. So the right span is short
    /// when moving briskly and long when crawling — which is exactly what a
    /// parallax target delivers automatically.
    ///
    /// 15 px is chosen to sit just above the observability floor: at f=537 px
    /// and 5 m range that is roughly 4-5 frames of a walking pace, and tens of
    /// frames when nearly stationary. The original 60 px produced ~12-frame
    /// spans at walking pace, deep into the regime where frame-to-frame won.
    /// Confirm this on hardware with the A/B in the README.
    double keyframeMaxParallaxPx{15.0};
    /// ...or unconditionally after this many frames, to bound linearisation error.
    int keyframeMaxAgeFrames{30};

    // ---- Motion plausibility ---------------------------------------------
    /// Reject solutions implying motion no rigid platform could produce.
    /// These are per-keyframe-span, not per-frame, so they are generous.
    double maxTranslationPerSpanM{5.0};
    double maxRotationPerSpanRad{1.5};

    // ---- Measurement noise model -----------------------------------------
    /// Std dev of the pixel measurement, used to scale the output covariance.
    double pixelSigmaPx{0.5};
    /// Std dev of the disparity measurement.
    ///
    /// Careful: this is the block matcher's *noise*, not its 1/16 px subpixel
    /// quantisation step. Using the quantisation step here understates depth
    /// error by roughly an order of magnitude (it turns 31 cm at 5 m into a
    /// fictitious 4 cm). Measure it in Phase 0 against a flat wall at known
    /// range; 0.5 px is a realistic starting point.
    double disparitySigmaPx{0.5};

    /// Multiplier applied to the formal covariance before publishing.
    ///
    /// The formal (J^T W J)^-1 covariance assumes independent Gaussian
    /// measurements and a perfectly clean inlier set. Neither holds: keyframe
    /// triangulation error is correlated with the current-frame measurement
    /// through the shared feature, and RANSAC leaves some contamination. On
    /// synthetic scenes the empirical scatter came out ~2.7x the formal
    /// prediction, so the raw value is over-confident — the dangerous
    /// direction for a downstream EKF. 3.0 makes it mildly conservative.
    /// Re-derive this from real data once Phase 3 logs exist.
    double covarianceInflation{3.0};

    // ---- Motion model -----------------------------------------------------
    /// Seed the solve from the previous inter-frame motion rather than identity.
    bool useConstantVelocitySeed{true};
    /// Discard the velocity seed if the last estimate is older than this many
    /// frames — a stale prediction is worse than identity.
    int maxSeedAgeFrames{3};
};

}  // namespace oak_vio
