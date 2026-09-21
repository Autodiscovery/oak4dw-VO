#include "oak_vio/imu_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oak_vio {

ImuIntegrator::ImuIntegrator(ImuParams params) : params_(std::move(params)) {}

void ImuIntegrator::setParams(const ImuParams& params) {
    params_ = params;
}

void ImuIntegrator::setImuToCameraRotation(const Mat3& rotation) {
    imuToCamera_ = rotation;
    // Calibration matrices arrive as floats through a JSON round trip, so they
    // are near-orthonormal rather than orthonormal. Left alone, the error
    // compounds through every integration step.
    Pose holder;
    holder.R = imuToCamera_;
    holder.normalise();
    imuToCamera_ = holder.R;
}

void ImuIntegrator::addSample(const ImuSample& sample) {
    ++totalSamples_;

    if(sample.hasAccel) {
        lastAccelImu_ = sample.acceleration;
        lastAccelCamera_ = imuToCamera_ * sample.acceleration;
        haveAccel_ = true;
    }

    if(!sample.hasGyro) {
        return;
    }

    // Device timestamps are monotonic per sensor. A sample that goes backwards
    // means the two clocks have been mixed somewhere, and integrating it would
    // subtract rotation. Drop it and count it, so the health line can say so.
    if(!buffer_.empty() && sample.timestampSeconds <= buffer_.back().timestampSeconds) {
        ++droppedOutOfOrder_;
        return;
    }

    updateBias(sample);

    buffer_.push_back(sample);

    const double cutoff = sample.timestampSeconds - params_.bufferSeconds;
    while(!buffer_.empty() && buffer_.front().timestampSeconds < cutoff) {
        buffer_.pop_front();
    }
}

void ImuIntegrator::updateBias(const ImuSample& sample) {
    if(!params_.estimateGyroBias) {
        return;
    }

    // Both tests are required. A platform in a smooth constant-rate turn holds
    // |a| at gravity while the gyro is plainly not zero, and a platform in free
    // fall or under linear acceleration reads a near-zero gyro with the wrong
    // |a|. Either test alone therefore admits motion as stillness, which would
    // learn a bias that is really a real rotation rate and then subtract that
    // rate from every subsequent frame.
    const bool gyroStill = sample.angularVelocity.norm() < params_.staticGyroThreshold;
    const Vec3 accel = sample.hasAccel ? sample.acceleration : lastAccelImu_;
    const bool haveAccelForTest = sample.hasAccel || haveAccel_;
    const bool accelStill = haveAccelForTest && std::abs(accel.norm() - params_.gravityMagnitude) < params_.staticAccelTolerance;

    if(!gyroStill || !accelStill) {
        return;
    }

    if(staticSamples_ == 0) {
        gyroBias_ = sample.angularVelocity;
    } else if(staticSamples_ < params_.biasMinSamples) {
        // Plain running mean while warming up: it converges in the number of
        // samples we are already waiting for, where exponential forgetting
        // would still be most of the way back at its starting value.
        const double n = static_cast<double>(staticSamples_) + 1.0;
        gyroBias_ += (sample.angularVelocity - gyroBias_) / n;
    } else {
        // Then switch to forgetting, so the estimate can follow thermal drift
        // instead of being anchored by the first second of the session.
        const double f = std::clamp(params_.biasForgettingFactor, 0.0, 1.0);
        gyroBias_ = f * gyroBias_ + (1.0 - f) * sample.angularVelocity;
    }

    if(staticSamples_ < params_.biasMinSamples) {
        ++staticSamples_;
    }
}

RotationPrior ImuIntegrator::deltaRotation(double t0, double t1) const {
    RotationPrior prior;

    if(!params_.enabled) {
        prior.rejection = "IMU disabled";
        return prior;
    }
    const double window = t1 - t0;
    if(!(window > 0.0)) {
        prior.rejection = "non-positive frame interval";
        return prior;
    }
    if(buffer_.size() < 2) {
        prior.rejection = "fewer than two gyro samples buffered";
        return prior;
    }

    const Vec3 bias = biasReady() ? gyroBias_ : Vec3::Zero();
    const double nominalStep = params_.reportRateHz > 0.0 ? 1.0 / params_.reportRateHz : 0.0;
    const double maxCountedGap = std::max(nominalStep * params_.maxSampleGapFactor, 1e-9);

    // Angular velocity is held piecewise-linear between samples and integrated
    // exactly over each sub-interval clipped to the window, so the answer does
    // not depend on where the samples happen to fall relative to the frame
    // boundaries. Sub-interval rotations compose by RIGHT multiplication,
    // matching the body-frame convention documented in the header.
    Mat3 q = Mat3::Identity();
    double covered = 0.0;
    int used = 0;

    // Rotate into the camera frame before integrating. This is exact, not an
    // approximation: R exp([w]x dt) R^T == exp([R w]x dt).
    const auto rateAt = [&](const ImuSample& sample) { return imuToCamera_ * (sample.angularVelocity - bias); };

    // ---- Leading edge ----------------------------------------------------
    // A frame timestamp almost never lands exactly on a sample, so the window
    // typically opens up to one sample interval before the first sample. Left
    // unhandled, that missing sliver is a SYSTEMATIC shortfall — always in the
    // same direction, and proportional to the rotation rate — which is exactly
    // the kind of error that survives review because the prior still looks
    // roughly right. Holding the first sample's rate across it costs nothing
    // and removes the bias; the gap is only counted as covered if it is short
    // enough to be sample alignment rather than a real dropout.
    {
        const ImuSample& first = buffer_.front();
        const double a = t0;
        const double b = std::min(first.timestampSeconds, t1);
        if(b > a) {
            q = q * expSO3(rateAt(first) * (b - a));
            if(b - a <= maxCountedGap) {
                covered += b - a;
            }
            ++used;
        }
    }

    for(std::size_t i = 0; i + 1 < buffer_.size(); ++i) {
        const ImuSample& s0 = buffer_[i];
        const ImuSample& s1 = buffer_[i + 1];
        const double span = s1.timestampSeconds - s0.timestampSeconds;
        if(span <= 0.0) {
            continue;
        }

        const double a = std::max(s0.timestampSeconds, t0);
        const double b = std::min(s1.timestampSeconds, t1);
        if(b <= a) {
            continue;  // sub-interval lies entirely outside the window
        }

        const Vec3 w0 = rateAt(s0);
        const Vec3 w1 = rateAt(s1);

        const double alphaA = (a - s0.timestampSeconds) / span;
        const double alphaB = (b - s0.timestampSeconds) / span;
        const Vec3 wa = w0 + (w1 - w0) * alphaA;
        const Vec3 wb = w0 + (w1 - w0) * alphaB;

        q = q * expSO3(0.5 * (wa + wb) * (b - a));
        if(span <= maxCountedGap) {
            covered += b - a;
        }
        ++used;
    }

    // ---- Trailing edge ---------------------------------------------------
    // The mirror image, and the more likely of the two in practice: the frame
    // callback runs as soon as the image arrives, so the IMU queue is usually a
    // sample or two behind the frame timestamp.
    {
        const ImuSample& last = buffer_.back();
        const double a = std::max(last.timestampSeconds, t0);
        const double b = t1;
        if(b > a) {
            q = q * expSO3(rateAt(last) * (b - a));
            if(b - a <= maxCountedGap) {
                covered += b - a;
            }
            ++used;
        }
    }

    prior.samplesUsed = used;
    prior.coverage = covered / window;

    if(used == 0) {
        prior.rejection = "no gyro samples inside the frame interval";
        return prior;
    }

    // A window missing a chunk of its samples has dropped a transport burst,
    // and the integration silently attributes zero rotation to the gap. Better
    // to fall back to the constant-velocity seed than to seed RANSAC with a
    // rotation that is confidently too small.
    if(prior.coverage < params_.minSampleCoverage) {
        prior.rejection = "gyro coverage of the frame interval too low";
        return prior;
    }

    Pose holder;
    holder.R = q;
    holder.normalise();

    // The transpose: q integrates world-from-body, the estimator wants
    // current-from-previous. See the header.
    prior.deltaRotation = holder.R.transpose();
    prior.angleRad = std::abs(logSO3(prior.deltaRotation).norm());

    if(prior.angleRad > params_.maxRotationPerIntervalRad) {
        prior.rejection = "implausible rotation over one frame interval";
        return prior;
    }

    prior.valid = true;
    return prior;
}

double ImuIntegrator::bufferedSpanSeconds() const {
    if(buffer_.size() < 2) {
        return 0.0;
    }
    return buffer_.back().timestampSeconds - buffer_.front().timestampSeconds;
}

void ImuIntegrator::reset() {
    buffer_.clear();
}

void ImuIntegrator::resetBias() {
    gyroBias_.setZero();
    staticSamples_ = 0;
}

Mat3 blendRotation(const Mat3& from, const Mat3& to, double weight) {
    const double w = std::clamp(weight, 0.0, 1.0);
    if(w <= 0.0) {
        return from;
    }
    if(w >= 1.0) {
        return to;
    }
    // Geodesic interpolation on SO(3). Interpolating the matrices elementwise
    // would leave SO(3) and need re-orthonormalising, which is not the same
    // curve and is not a rotation of the intended magnitude.
    Pose holder;
    holder.R = from * expSO3(logSO3(from.transpose() * to) * w);
    holder.normalise();
    return holder.R;
}

}  // namespace oak_vio
