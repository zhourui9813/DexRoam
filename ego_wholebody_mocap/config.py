from __future__ import annotations

# Base directory for capture output. Relative paths are resolved from project root.
OUTPUT_DIR = "data/test_data"

# Target VR image push rate in FPS. Use 0 to push as fast as possible.
VR_PUSH_FPS = 20.0

# VR image plane height in Vuer world units.
VR_IMAGE_HEIGHT = 4.0

# VR image plane position [x, y, z] in Vuer world units.
VR_IMAGE_POSITION = (0.0, -1.0, 3)

# ZED camera resolution. Options: VGA, HD720, HD1080, HD2K.
ZED_RES = "HD1080"

# ZED camera capture FPS.
CAMERA_CAPTURE_FPS = 30

# Image source shown in VR. Options: left, right, both.
STREAM_MODE = "left"

# Maximum raw uint8 ZED frame backlog before aborting the active episode, in GiB.
BACKLOG_BUDGET_GIB = 8.0

# Number of raw frame slots allocated per shared-memory slab.
FRAME_SLOTS_PER_SLAB = 32

# Number of shared-memory slabs allocated when each episode starts.
INITIAL_FRAME_SLABS = 2

# JPEG quality used when ZED storage mode is jpeg. Valid OpenCV range: 0-100.
JPEG_QUALITY = 85

# ZED HDF5 storage mode. Options: "jpeg", "raw_uint8".
ZED_STORAGE_MODE = "jpeg"

# Writer abort code: no abort requested.
ABORT_NONE = 0

# Writer abort code: raw frame backlog exceeded the configured budget.
ABORT_OVERFLOW = 1

# Writer abort code: discard active episode without keeping pending output.
ABORT_DISCARD = 2

# Duration for transient status feedback text in the VR overlay, in seconds.
STATUS_FEEDBACK_SECONDS = 1.5
