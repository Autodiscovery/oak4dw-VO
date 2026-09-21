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
uint32 num_observations     # tracked features arriving from the front-end
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

# --- Front-end ---
# Which tracker produced num_observations. Worth publishing rather than
# inferring from parameters: the hardware block is optional and can fail over to
# nothing at all, and the CPU cost below only applies to one of them.
uint8 FRONTEND_CPU      = 0
uint8 FRONTEND_HARDWARE = 1
uint8 frontend
float32 tracker_ms          # front-end cost; ~0 on the hardware path

# --- Depth source ---
uint8 DEPTH_BLOCK_MATCHER = 0
uint8 DEPTH_NEURAL        = 1   # LENS; smoothed and partly inferred
uint8 depth_source

# --- IMU (Phase 4) ---
# gyro_prior_used says whether THIS frame's seed carried the gyro-integrated
# rotation. A prior that is silently rejected every frame is indistinguishable
# from one that does not help, so the rejection reason is published too.
bool imu_available
bool gyro_prior_used
float32 gyro_prior_angle_rad   # rotation the prior supplied, radians
string gyro_prior_rejection    # empty when the prior was used
bool gyro_bias_ready           # the static-interval bias estimate has converged
float32[3] gyro_bias           # rad/s, IMU frame

# True when ~/vo/odometry is coming from the error-state filter rather than
# straight from the VO integrator. The filter is a scaffold and off by default;
# this flag is how a consumer can tell which one it is reading.
bool imu_filter_active
