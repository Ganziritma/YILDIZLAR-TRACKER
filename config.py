# =============================================================================
# config.py — YILDIZLAR Tracker Configuration
# =============================================================================

# ── Detector selection ────────────────────────────────────────────────────────
# "yolo"        → faster, recommended for CPU-only machines
# "transformer" → RF-DETR, much more accurate but needs a GPU to be real-time
DETECTOR = "transformer"

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = "C:\\Users\\yusuf\\OneDrive - Yildiz Technical University\\Masaüstü\\YILDIZLAR_TRACKER\\models\\yolov8_avcı.pt"
TRANSFORMER_PATH = "C:\\Users\\yusuf\\OneDrive - Yildiz Technical University\\Masaüstü\\YILDIZLAR_TRACKER\\models\\checkpoint_best_ema.pth" # checkpoint_best_ema.pth or checkpoint_best_total
VIDEO_PATH       = "C:\\Users\\yusuf\\OneDrive - Yildiz Technical University\\Masaüstü\\YILDIZLAR_TRACKER\\videos\\deneme.mp4"
OUTPUT_PATH      = "C:\\Users\\yusuf\\Downloads\\tracker_out.mp4"

# ── YOLO settings ─────────────────────────────────────────────────────────────
# On CPU: EVERY_N=3 → ~10Hz at 30fps. Use imgsz=320 for max speed.
YOLO_EVERY_N        = 3
YOLO_IMGSZ          = 320    # 320=fastest on CPU, 480=balanced, 640=accurate
YOLO_CONF           = 0.19   # lowered slightly to catch more detections

# ── RF-DETR settings (only relevant when DETECTOR = "transformer") ────────────
# On CPU this will be ~1-2 FPS inference. Use EVERY_N=30 to not freeze the UI.
# On a GPU it can do ~15-30 FPS inference.
TRANSFORMER_EVERY_N     = 11   # benchmark: inf=371ms, 11 frames pass per inference
TRANSFORMER_IMGSZ       = 576  # must match training resolution
TRANSFORMER_CONF        = 0.49 # benchmark: 25th-pct of seen confs — cuts false positives
TRANSFORMER_NUM_CLASSES = 2

# ── Shared detection settings ─────────────────────────────────────────────────
YOLO_MAX_AREA_RATIO = 0.20

# ── ByteTrack ─────────────────────────────────────────────────────────────────
BT_MAX_AGE      = 90
BT_MATCH_THRESH = 0.35
BT_MIN_HITS      = 1
# track_thresh: detections above this confidence create new tracks.
# Must be BELOW your detector's typical output confidence.
# YOLO gives ~0.40 → set to 0.25. RF-DETR gives ~0.65+ → 0.25 also fine.
BT_TRACK_THRESH  = 0.19

# ── Primary target lock ───────────────────────────────────────────────────────
AUTO_LOCK               = True
SPATIAL_FALLBACK_DELAY  = 15
SPATIAL_FALLBACK_RADIUS = 0.20
DEADRECK_MAX_FRAMES     = 90

# ── Error vector ──────────────────────────────────────────────────────────────
ERROR_SMOOTH_ALPHA = 0.2

# ── HUD / Display ─────────────────────────────────────────────────────────────
TRAIL_LENGTH = 60


def detector_every_n() -> int:
    return TRANSFORMER_EVERY_N if DETECTOR == "transformer" else YOLO_EVERY_N