# =============================================================================
# hud.py — HUD / overlay drawing functions
# Pure drawing — no tracker logic here.
# =============================================================================

import cv2
import numpy as np
from typing import Optional
from collections import deque
import config


def draw_crosshair(frame: np.ndarray, cx: int, cy: int):
    pass  # removed


def draw_trail(frame: np.ndarray, trail: deque):
    pass  # removed


def draw_dead_reckoning(frame: np.ndarray, bbox: tuple, age: int):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    fade  = max(0.0, 1.0 - age / config.DEADRECK_MAX_FRAMES)
    color = (0, int(180 * fade), int(255 * fade))

    def dashed_line(p1, p2):
        x0, y0 = p1; xe, ye = p2
        length = int(np.hypot(xe - x0, ye - y0))
        if length == 0:
            return
        dx, dy = (xe - x0) / length, (ye - y0) / length
        pos, drawing = 0, True
        while pos < length:
            seg_len = 8 if drawing else 5
            end = min(pos + seg_len, length)
            if drawing:
                sx, sy = int(x0 + pos * dx), int(y0 + pos * dy)
                ex, ey = int(x0 + end * dx), int(y0 + end * dy)
                cv2.line(frame, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)
            pos, drawing = end, not drawing

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))
    cv2.putText(frame, f"COAST +{age}f",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_other_tracks(frame: np.ndarray, tracks: list, primary_id: Optional[int]):
    for t in tracks:
        if int(t[4]) == primary_id:
            continue
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid  = int(t[4])
        conf = float(t[5]) if t.shape[0] > 5 else 0.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(frame, f"#{tid} {conf:.2f}",
                    (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (80, 80, 80), 1, cv2.LINE_AA)


def draw_primary_track(frame: np.ndarray, track: np.ndarray,
                        raw_error: tuple, smooth_error: tuple,
                        approach_str: str,
                        velocity: Optional[tuple],
                        cx_f: int, cy_f: int,
                        frame_w: int, frame_h: int):
    x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
    tid  = int(track[4])
    conf = float(track[5]) if track.shape[0] > 5 else 0.0
    col  = (0, 230, 0)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # Corner brackets
    t = 12
    for px, py, sx, sy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + sx * t, py), col, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + sy * t), col, 2, cv2.LINE_AA)

    # Lock label above bbox
    label = f"LOCK #{tid}  conf:{conf:.2f}"
    if approach_str:
        label += f"  {approach_str}"
    cv2.putText(frame, label,
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray,
                    frame_w: int, frame_h: int,
                    fps: float, det_ms: float, inf_fps: float,
                    n_tracks: int, det_ran: bool,
                    is_locked: bool, is_coasting: bool,
                    primary_id: Optional[int],
                    frames_since_det: int,
                    smooth_error: tuple,
                    recording: bool):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if is_locked:
        if is_coasting:
            state = f"COASTING #{primary_id}  +{frames_since_det}f"
            scol  = (0, 200, 255)
        else:
            state = f"LOCKED #{primary_id}"
            scol  = (0, 230, 0)
    else:
        state = "NO LOCK"
        scol  = (0, 60, 220)

    det_tag = f"[{config.DETECTOR.upper()}]" if det_ran else "            "
    rec_tag = "  [REC]" if recording else ""

    cv2.putText(frame,
                f"disp:{fps:.0f}fps  inf:{inf_fps:.1f}fps  "
                f"det:{det_ms:.0f}ms  tracks:{n_tracks}  {det_tag}{rec_tag}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, state,
                (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, scol, 1, cv2.LINE_AA)

    cv2.putText(frame, "U=unlock  R=record  Q=quit",
                (8, frame_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1, cv2.LINE_AA)