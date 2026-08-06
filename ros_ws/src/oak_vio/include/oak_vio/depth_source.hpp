// Where feature depth comes from.
//
// The estimator only ever needs one thing from the depth pipeline: the
// disparity at a given rectified-left pixel. Putting that behind an interface
// keeps two doors open that the plan cares about:
//
//   * LENS / NeuralDepth can substitute for the HW block matcher, so VO can
//     ride on whichever depth engine is already running rather than forcing a
//     second one.
//   * If Phase 0 shows the rectified FoV is unusable and we fall back to
//     Kannala-Brandt on raw fisheye, a sparse left/right matcher can be
//     dropped in here without touching the estimator.
#pragma once

#include <cmath>
#include <cstdint>

namespace oak_vio {

class DepthSource {
   public:
    virtual ~DepthSource() = default;

    /// Disparity in pixels at the given rectified-left pixel, or a
    /// non-positive value when unavailable (invalid pixel, out of bounds,
    /// below confidence threshold).
    [[nodiscard]] virtual float disparityAt(float u, float v) const = 0;
};

/// Disparity straight from dai::node::StereoDepth.
///
/// RVC4 reports subpixel disparity as fixed point; `subpixelFractionalBits`
/// is the shift (3 bits = 1/8 px, 4 bits = 1/16 px, 5 bits = 1/32 px). It must
/// match StereoDepth's configured subpixel setting or every depth will be
/// scaled wrong by a power of two — a mistake that produces a VO trajectory
/// with plausible shape and badly wrong scale.
class DisparityMapSource final : public DepthSource {
   public:
    DisparityMapSource(const std::uint16_t* data, int width, int height, int stridePixels, int subpixelFractionalBits)
        : data_(data),
          width_(width),
          height_(height),
          stride_(stridePixels),
          scale_(1.0F / static_cast<float>(1U << static_cast<unsigned>(subpixelFractionalBits))) {}

    [[nodiscard]] float disparityAt(float u, float v) const override {
        const int x = static_cast<int>(std::lround(u));
        const int y = static_cast<int>(std::lround(v));
        if(data_ == nullptr || x < 0 || y < 0 || x >= width_ || y >= height_) {
            return -1.0F;
        }
        const std::uint16_t raw = data_[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride_) + static_cast<std::size_t>(x)];
        if(raw == 0) {
            return -1.0F;  // block matcher marks invalid pixels as zero
        }
        return static_cast<float>(raw) * scale_;
    }

   private:
    const std::uint16_t* data_{nullptr};
    int width_{0};
    int height_{0};
    int stride_{0};
    float scale_{1.0F};
};

/// Metric depth (millimetres) converted to equivalent disparity, for consuming
/// LENS / NeuralDepth output.
///
/// A caveat worth keeping in mind: a neural depth map is smoothed and, in
/// low-texture regions, partly inferred rather than measured. That is good for
/// dense perception and less good for VO, where a locally biased depth at a
/// feature translates directly into a biased pose. Prefer the block matcher
/// for VO when both are available; use this when the DSP path is the only one
/// running.
class DepthMapSource final : public DepthSource {
   public:
    DepthMapSource(const std::uint16_t* depthMillimetres, int width, int height, int stridePixels, double fxBaselineM)
        : data_(depthMillimetres), width_(width), height_(height), stride_(stridePixels), fxBaseline_(fxBaselineM) {}

    [[nodiscard]] float disparityAt(float u, float v) const override {
        const int x = static_cast<int>(std::lround(u));
        const int y = static_cast<int>(std::lround(v));
        if(data_ == nullptr || x < 0 || y < 0 || x >= width_ || y >= height_) {
            return -1.0F;
        }
        const std::uint16_t millimetres = data_[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride_) + static_cast<std::size_t>(x)];
        if(millimetres == 0) {
            return -1.0F;
        }
        const double metres = static_cast<double>(millimetres) * 1e-3;
        return static_cast<float>(fxBaseline_ / metres);
    }

   private:
    const std::uint16_t* data_{nullptr};
    int width_{0};
    int height_{0};
    int stride_{0};
    double fxBaseline_{0.0};
};

}  // namespace oak_vio
