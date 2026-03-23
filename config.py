"""Shared configuration constants for discopackage frame extraction pipeline."""

N_MAIN = 100
N_NEUTRAL = 40
N_TOTAL = N_MAIN + N_NEUTRAL

# seconds to skip at start (titles)
SKIP_START = 180

# seconds to skip at end (credits)
SKIP_END = 420

CLIP_DUP_THRESHOLD = 0.93

FACE_MIN_SIZE = 40
FACE_MAX_YAW = 45.0
FACE_MIN_CONFIDENCE = 0.5
FACE_MIN_SHARPNESS = 20.0

DBSCAN_EPS = 20.0
DBSCAN_MIN_SAMPLES = 3

QUALITY_KEEP_RATIO = 0.85

# Diverse CLIP pool size = POOL_MULTIPLIER * (N_MAIN + N_NEUTRAL), capped by available frames.
POOL_MULTIPLIER = 4

