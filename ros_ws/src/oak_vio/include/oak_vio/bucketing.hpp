// Spatial bucketing of correspondences.
//
// Without this, a single richly textured object close to the camera can supply
// most of the correspondences and dominate the fit, which both biases the
// solution and leaves the geometry poorly conditioned. Capping features per
// grid cell forces the solve to draw on the whole image.
//
// It matters more on this camera than on a narrow lens: with a ~100 deg
// rectified FoV, the periphery carries most of the rotational information, and
// an unbucketed feature set tends to clump wherever the texture happens to be.
#pragma once

#include <vector>

#include "oak_vio/types.hpp"
#include "oak_vio/vio_params.hpp"

namespace oak_vio {

/// Indices into `correspondences`, spread across a bucketCols x bucketRows
/// grid over the current image, at most maxFeaturesPerBucket per cell.
///
/// Within a cell, features are ranked by age (longer-lived tracks are more
/// reliable) and then by tracking error. Bucketing is done on the *current*
/// frame position, since that is where we care about spatial coverage.
[[nodiscard]] std::vector<int> selectBucketed(const std::vector<Correspondence>& correspondences,
                                              int imageWidth,
                                              int imageHeight,
                                              const VioParams& params);

/// Median parallax over the given subset, used by the keyframe promotion
/// policy. Returns 0 for an empty selection.
[[nodiscard]] double medianParallax(const std::vector<Correspondence>& correspondences, const std::vector<int>& indices);

}  // namespace oak_vio
