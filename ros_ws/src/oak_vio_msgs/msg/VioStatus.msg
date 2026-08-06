# Health of the on-device stereo VO estimator, one message per processed frame.
#
# Publish rate matches the VO frame rate. Consume this rather than inferring
# health from the odometry topic: when tracking is lost the estimator keeps
# publishing the last pose with an inflated covariance, so /odometry alone
# looks superficially fine.

std_msgs/Header header

# --- Tracking state ---
uint8 INITIALISING = 0
uint8 TRACKING     = 1
uint8 LOW_INLIERS  = 2   # solved, but weak support; covariance is inflated
uint8 LOST         = 3   # no usable solution; pose held, do not integrate
uint8 state

# Human-readable reason for the current state, or the keyframe promotion
# reason. Empty when nominal.
string note

# --- Feature accounting, in pipeline order ---
uint32 num_observations     # tracked features arriving from the HW tracker
uint32 num_correspondences  # survived keyframe matching and gating
uint32 num_selected         # survived spatial bucketing
uint32 num_inliers          # survived RANSAC
float32 inlier_ratio        # num_inliers / num_selected

# --- Keyframe ---
bool keyframe_promoted      # true on the frame a new keyframe was anchored
int64 keyframe_index        # frame index of the current reference keyframe
float32 median_parallax_px  # median feature parallax from the keyframe

# --- Cost ---
float32 solve_ms            # wall-clock time in the estimator for this frame
int64 frame_index
