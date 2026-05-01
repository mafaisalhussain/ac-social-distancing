"""
config.py — Settings for AC Social Distancing Detector
"""

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── poseNet ───────────────────────────────────────────────────────────────────
# "resnet18-body"  = faster  (recommended for Jetson Nano)
# "resnet50-body"  = more accurate but slower
POSENET_MODEL      = "resnet18-body"
KEYPOINT_THRESHOLD = 0.15   # lower = detect more keypoints, higher = stricter

# ── Social Distancing ─────────────────────────────────────────────────────────
# Pixel distance below which two people are flagged as a violation.
# Increase this number if people far apart are still flagging.
# Decrease if only very close people should trigger.
DISTANCE_THRESHOLD = 350   # pixels

# ── poseNet keypoint indices (resnet18-body COCO 18-point) ────────────────────
NOSE          = 0
LEFT_EYE      = 1
RIGHT_EYE     = 2
LEFT_EAR      = 3
RIGHT_EAR     = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER= 6
LEFT_ELBOW    = 7
RIGHT_ELBOW   = 8
LEFT_WRIST    = 9
RIGHT_WRIST   = 10
LEFT_HIP      = 11
RIGHT_HIP     = 12
LEFT_KNEE     = 13
RIGHT_KNEE    = 14
LEFT_ANKLE    = 15
RIGHT_ANKLE   = 16
NECK          = 17

# Skeleton pairs to draw
SKELETON_PAIRS = [
    (NECK, LEFT_SHOULDER),
    (NECK, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (NOSE, NECK),
]

# ── AC HUD Colors (BGR for OpenCV) ────────────────────────────────────────────
COLOR_GOLD      = (60,  180, 210)
COLOR_SAFE      = (60,  200,  60)
COLOR_VIOLATION = (0,   40,  220)
COLOR_SKELETON  = (80,  160, 210)
COLOR_HUD_TEXT  = (180, 220, 240)
COLOR_GOLD_DIM  = (40,  110, 130)
