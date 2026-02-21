# =============================================================================
# main.py — YILDIZLAR Tracker entry point
#
# Usage:
#   python main.py                        # uses VIDEO_PATH from config.py
#   python main.py path/to/video.mp4      # open a specific file
#   python main.py path/to/folder/        # pick from all videos in folder
# =============================================================================

import cv2
import numpy as np
import time
import threading
import sys
import os
from pathlib import Path
from typing import Optional

import config
from detector import build_detector
from tracker_core import AirCombatTracker
import hud

# ── Video path resolution ─────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}

def pick_video_from_folder(folder: str) -> str:
    """List all video files in folder and let the user pick one."""
    folder = Path(folder)
    videos = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])
    if not videos:
        raise RuntimeError(f"No video files found in: {folder}")

    print(f"\nVideos in {folder}:")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {v.name}")
    print()

    while True:
        try:
            choice = input(f"Select video [1-{len(videos)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return str(videos[idx])
            print(f"  Please enter a number between 1 and {len(videos)}.")
        except (ValueError, EOFError):
            print("  Invalid input.")

def resolve_video_path() -> str:
    """
    Decide which video to open:
      - sys.argv[1] is a file  → use it directly
      - sys.argv[1] is a dir   → list videos, ask user to pick
      - no arg                 → fall back to config.VIDEO_PATH
    """
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            return pick_video_from_folder(arg)
        elif os.path.isfile(arg):
            return arg
        else:
            raise RuntimeError(f"Path not found: {arg}")
    return config.VIDEO_PATH

# ── Resolve video before anything else ───────────────────────────────────────
video_path = resolve_video_path()
print(f"\nOpening: {video_path}")

# ── Threaded detector wrapper ─────────────────────────────────────────────────
# Runs inference in a background thread so the display loop never freezes.
# The main loop submits a frame for detection and immediately continues
# drawing / recording; the result appears on the next check.

class AsyncDetector:
    """
    Wraps a detector so inference runs in a background thread.

    Usage:
        async_det = AsyncDetector(detector)
        async_det.submit(frame)          # fire and forget
        dets, is_fresh = async_det.get() # always returns immediately
    """
    def __init__(self, det):
        self._det      = det
        self._result   = np.empty((0, 6), dtype=np.float32)
        self._fresh    = False       # True for exactly one get() call after result lands
        self._busy     = False       # inference running right now
        self._det_ms   = 0.0
        self._lock     = threading.Lock()

    def submit(self, frame: np.ndarray) -> bool:
        """
        Submit frame for async detection.
        Returns True if submitted, False if still busy with previous frame.
        """
        with self._lock:
            if self._busy:
                return False
            self._busy  = True
            self._fresh = False

        # Copy the frame so the main thread can keep modifying its own copy
        t = threading.Thread(target=self._run, args=(frame.copy(),), daemon=True)
        t.start()
        return True

    def _run(self, frame: np.ndarray):
        t0 = time.perf_counter()
        result = self._det(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._result  = result
            self._fresh   = True
            self._busy    = False
            self._det_ms  = ms

    def get(self):
        """
        Returns (detections, is_fresh, det_ms).
        is_fresh is True only the FIRST time you call get() after a result lands.
        """
        with self._lock:
            fresh       = self._fresh
            self._fresh = False          # consume freshness
            return self._result.copy(), fresh, self._det_ms

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy


# ── Setup ─────────────────────────────────────────────────────────────────────
detector     = build_detector()
async_det    = AsyncDetector(detector)
every_n      = config.detector_every_n()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {video_path}")

native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracker = AirCombatTracker(frame_w, frame_h)
cx_f, cy_f = frame_w // 2, frame_h // 2

print(f"Detector : {detector.name}  (runs every {every_n} frames, async)")
print(f"Video    : {frame_w}x{frame_h} @ {native_fps:.1f} fps")
print("U=unlock  R=record  Q=quit\n")

# ── Recording ─────────────────────────────────────────────────────────────────
recording : bool             = False
writer    : Optional[object] = None

def start_recording():
    global writer, recording
    # Save next to the input video, with _tracked suffix
    stem   = Path(video_path).stem
    folder = Path(video_path).parent
    out    = str(folder / f"{stem}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, native_fps, (frame_w, frame_h))
    recording = True
    print(f"[REC] Started → {out}")

def stop_recording():
    global writer, recording
    if writer:
        writer.release()
        writer = None
    recording = False
    print("[REC] Stopped.")

# ── State ─────────────────────────────────────────────────────────────────────
frame_count  : int   = 0
det_ms       : float = 0.0
det_ran      : bool  = False
fps_display  : float = 0.0
inf_fps      : float = 0.0
t_prev       : float = time.perf_counter()

# Latest confirmed detections from async thread
latest_dets  : np.ndarray = np.empty((0, 6), dtype=np.float32)

cv2.namedWindow("YILDIZLAR Tracker")

# ── Debug overlay state ───────────────────────────────────────────────────────
debug_mode : bool = False   # toggle with D key

# ── Debug overlay drawing ─────────────────────────────────────────────────────
def _draw_debug_overlay(frame: np.ndarray, tracker, frame_w: int, frame_h: int):
    """
    Draws rich diagnostic info on top of the frame.
    Toggle with D key. Does not affect recording output.
    """
    pt_color = (0, 255, 0) if tracker.flow_healthy else (0, 80, 255)
    if tracker.flow_pts_display is not None:
        for pt in tracker.flow_pts_display:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, pt_color, -1)

    if tracker.last_bbox is not None:
        x1, y1, x2, y2 = tracker.last_bbox
        for (p1, p2) in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                         ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
            pts_line = np.linspace(0, 1, 20)
            for i in range(0, len(pts_line)-1, 2):
                sx = int(p1[0] + pts_line[i]   * (p2[0]-p1[0]))
                sy = int(p1[1] + pts_line[i]   * (p2[1]-p1[1]))
                ex = int(p1[0] + pts_line[i+1] * (p2[0]-p1[0]))
                ey = int(p1[1] + pts_line[i+1] * (p2[1]-p1[1]))
                cv2.line(frame, (sx,sy), (ex,ey), (0,220,220), 1)
        cv2.putText(frame, "LAST_DET", (x1, max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,220,220), 1)

    if tracker.flow_healthy and tracker._flow_bbox is not None:
        x1, y1, x2, y2 = tracker._flow_bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,220,0), 1)
        cv2.putText(frame, "FLOW", (x1, y2+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,220,0), 1)

    bar_w, bar_h = 120, 14
    bx = frame_w - bar_w - 10
    by = 55
    cv2.rectangle(frame, (bx, by), (bx+bar_w, by+bar_h), (50,50,50), -1)
    fill = int(bar_w * tracker.flow_health)
    bar_col = (0,230,0) if tracker.flow_health > 0.6 else \
              (0,165,255) if tracker.flow_health > 0.3 else (0,60,220)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx+fill, by+bar_h), bar_col, -1)
    cv2.rectangle(frame, (bx, by), (bx+bar_w, by+bar_h), (120,120,120), 1)
    cv2.putText(frame, f"FLOW {tracker.flow_health:.0%}",
                (bx, by-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)

    panel_y = frame_h - 72
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (frame_w, frame_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def dp(text, row, col=(160,160,160)):
        cv2.putText(frame, text, (8, panel_y + 16 + row*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    dp(f"STATUS: {tracker.debug_status}", 0, (0,230,0) if tracker.is_locked else (0,60,220))
    dp(f"OF: pts_alive={tracker._of_pts_alive}  "
       f"fail_streak={tracker._of_fail_streak}/3  "
       f"healthy={'YES' if tracker.flow_healthy else 'NO'}  "
       f"match_ref={tracker.match_ref_source}  "
       f"last_jump={tracker.last_match_jump:.0f}px", 1,
       (0,230,0) if tracker.flow_healthy else (0,80,255))
    dp(f"det_miss={tracker._det_miss_count}/{tracker._MAX_DET_MISSES}  "
       f"frames_since_det={tracker.frames_since_det}  "
       f"last_conf={tracker._last_conf:.2f}  "
       f"locked={'YES' if tracker.is_locked else 'NO'}", 2)
    dp("D=debug  U=unlock  R=record  Q=quit", 3, (60,60,60))


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % every_n == 0:
        async_det.submit(frame)

    fresh_dets, is_fresh, _ms = async_det.get()
    if is_fresh:
        latest_dets = fresh_dets
        det_ms      = _ms
        det_ran     = True
        if det_ms > 0:
            inf_fps = 0.2 * (1000.0 / det_ms) + 0.8 * inf_fps
    else:
        det_ran = False

    confirmed = tracker.update(
        latest_dets if det_ran else np.empty((0, 6), dtype=np.float32),
        frame,
        det_ran=det_ran,
    )
    if det_ran:
        latest_dets = np.empty((0, 6), dtype=np.float32)

    hud.draw_trail(frame, tracker.trail)

    if tracker.is_coasting and not confirmed and tracker.last_bbox is not None:
        hud.draw_dead_reckoning(frame, tracker.last_bbox, tracker.frames_since_det)

    hud.draw_other_tracks(frame, tracker.all_tracks, tracker.primary_id)

    primary_row = None
    for t in tracker.all_tracks:
        if int(t[4]) == tracker.primary_id:
            primary_row = t
            break

    if primary_row is not None:
        hud.draw_primary_track(
            frame, primary_row,
            tracker.raw_error, tracker.smooth_error,
            tracker.approach_str, tracker.velocity,
            cx_f, cy_f, frame_w, frame_h
        )

    hud.draw_crosshair(frame, cx_f, cy_f)

    if debug_mode:
        _draw_debug_overlay(frame, tracker, frame_w, frame_h)

    hud.draw_status_bar(
        frame, frame_w, frame_h,
        fps_display, det_ms, inf_fps,
        len(tracker.all_tracks), det_ran,
        tracker.is_locked, tracker.is_coasting,
        tracker.primary_id, tracker.frames_since_det,
        tracker.smooth_error, recording
    )

    now    = time.perf_counter()
    dt     = now - t_prev
    t_prev = now
    if dt > 0:
        fps_display = 0.1 * (1.0 / dt) + 0.9 * fps_display

    if recording and writer:
        writer.write(frame)

    wait_ms = max(1, int(1000.0 / native_fps - (time.perf_counter() - now) * 1000))
    cv2.imshow("YILDIZLAR Tracker", frame)
    frame_count += 1

    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        tracker.unlock()
    elif key == ord('r'):
        stop_recording() if recording else start_recording()
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"[DEBUG] overlay {'ON' if debug_mode else 'OFF'}")

if recording:
    stop_recording()
cap.release()
cv2.destroyAllWindows()
print("Done.")