# =============================================================================
# tracker_core.py — Optical Flow Tracker with Flow Trust System + GMC
#
# FLOW TRUST SYSTEM:
#   _of_fail_streak  — consecutive frames where <MIN_FLOW_POINTS survived
#   flow_healthy     — True when fail_streak < threshold
#
# GLOBAL MOTION COMPENSATION (GMC):
#   Tracks background FAST keypoints to estimate camera ego-motion each frame.
#   Used for:
#     1. Match reference when flow is dead: warp last_bbox by camera motion
#     2. Flow-on-background detection (see below)
#
# FLOW-ON-BACKGROUND DETECTION (FIXED):
#   Previous version compared magnitudes only — fired constantly when both
#   flow and camera were near zero (static scene). Now uses VECTOR comparison:
#     - Decompose camera motion into (tx, ty) from GMC warp
#     - Compare to flow vector (vx, vy) per-component
#     - Only flag ON-BG if camera is actually moving (cam_disp > MIN_CAM_DISP)
#       AND flow vector closely matches camera motion vector
#   This way a static aircraft on a moving camera → flagged correctly.
#   A static aircraft on a static camera → NOT flagged (both zero is fine).
#
# JUMP THRESHOLDS:
#   Increased significantly to account for asynchronous detector time-travel delay.
# =============================================================================

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque
import config


LK_PARAMS = dict(
    winSize  = (31, 31),  # Increased from (15,15) to track faster pixel motion
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners   = 50,
    qualityLevel = 0.005,
    minDistance  = 3,
    blockSize    = 3,
)

MIN_FLOW_POINTS = 2
OF_FAIL_LIMIT   = 3

# Seed quality threshold
FLOW_STRONG_MIN_SEEDS = 10

# Jump thresholds per reference source (Increased to account for async detector lag)
MAX_JUMP_FLOW_STRONG = 300   
MAX_JUMP_FLOW_WEAK   = 150
MAX_JUMP_GMC         = 300
MAX_JUMP_FLOW_DEAD   = 500   

# Low-conf + large-jump gate
LOW_CONF_THRESH   = 0.55
LOW_CONF_MAX_JUMP = 120

# Flow-on-background detection:
#   Only active when camera is actually moving meaningfully.
#   When cam_disp < MIN_CAM_DISP we cannot distinguish target-on-camera
#   from background — don't flag ON-BG.
MIN_CAM_DISP         = 4.0   # px — below this, skip ON-BG check
FLOW_BG_VEC_THRESH_X = 6.0   # px — per-axis match tolerance
FLOW_BG_VEC_THRESH_Y = 6.0


# =============================================================================
# Global Motion Compensator
# =============================================================================

class GlobalMotionCompensator:
    """
    Estimates camera ego-motion each frame using background FAST keypoints.
    Exposes:
      warp     — 2x3 partial affine (rotation + translation)
      valid    — warp was successfully estimated this frame
      cam_tx   — camera translation X this frame (px)
      cam_ty   — camera translation Y this frame (px)
      cam_disp — camera translation magnitude (px)
    """

    _LK = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )

    def __init__(self):
        self._prev_gray = None
        self._fast      = cv2.FastFeatureDetector_create(threshold=20)
        self.warp       = np.eye(2, 3, dtype=np.float32)
        self.valid      = False
        self.cam_tx     = 0.0
        self.cam_ty     = 0.0
        self.cam_disp   = 0.0

    def update(self, gray: np.ndarray, target_bbox=None):
        if self._prev_gray is None:
            self._prev_gray = gray
            self._reset_state()
            return

        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        if target_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in target_bbox]
            pad = 30
            mask[max(0, y1-pad):min(gray.shape[0], y2+pad),
                 max(0, x1-pad):min(gray.shape[1], x2+pad)] = 0

        kps = self._fast.detect(self._prev_gray, mask)
        if len(kps) < 8:
            self._prev_gray = gray
            self._reset_state()
            return

        prev_pts = np.array([kp.pt for kp in kps],
                            dtype=np.float32).reshape(-1, 1, 2)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **self._LK
        )
        if status is None:
            self._prev_gray = gray
            self._reset_state()
            return

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]
        if len(good_prev) < 6:
            self._prev_gray = gray
            self._reset_state()
            return

        warp, _ = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
        )
        if warp is None:
            self._prev_gray = gray
            self._reset_state()
        else:
            self.warp     = warp.astype(np.float32)
            self.cam_tx   = float(warp[0, 2])
            self.cam_ty   = float(warp[1, 2])
            self.cam_disp = float(np.hypot(self.cam_tx, self.cam_ty))
            self.valid    = True
            self._prev_gray = gray

    def _reset_state(self):
        self.warp     = np.eye(2, 3, dtype=np.float32)
        self.valid    = False
        self.cam_tx   = 0.0
        self.cam_ty   = 0.0
        self.cam_disp = 0.0

    def compensate_point(self, x: float, y: float) -> Tuple[float, float]:
        pt     = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.transform(pt, self.warp)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    def compensate_bbox(self, bbox: Tuple) -> Tuple:
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        warped  = [self.compensate_point(x, y) for x, y in corners]
        xs = [p[0] for p in warped]
        ys = [p[1] for p in warped]
        return (min(xs), min(ys), max(xs), max(ys))

    def reset(self):
        self._prev_gray = None
        self._reset_state()


# =============================================================================
# Main tracker
# =============================================================================

class AirCombatTracker:

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._cx     = frame_w / 2.0
        self._cy     = frame_h / 2.0

        # ── Public state ───────────────────────────────────────────────────────
        self.primary_id       : Optional[int]             = None
        self.last_bbox        : Optional[Tuple]           = None
        self.frames_since_det : int                       = 0
        self._last_conf       : float                     = 0.0

        self.raw_error    : Tuple[float, float]           = (0.0, 0.0)
        self.smooth_error : Tuple[float, float]           = (0.0, 0.0)
        self._smooth_state: list                          = [0.0, 0.0]
        self.velocity     : Optional[Tuple[float, float]] = None
        self.approach_str : str                           = ""
        self.trail        : deque = deque(maxlen=config.TRAIL_LENGTH)
        self.all_tracks   : list  = []

        # ── Optical flow state ─────────────────────────────────────────────────
        self._prev_gray      : Optional[np.ndarray]          = None
        self._flow_pts       : Optional[np.ndarray]          = None
        self._flow_bbox      : Optional[Tuple]               = None
        self._of_box_hw      : Optional[Tuple[float, float]] = None
        self._of_fail_streak : int  = 0
        self._of_pts_alive   : int  = 0
        self._seed_count     : int  = 0
        self._flow_on_bg     : bool = False

        # ── GMC ────────────────────────────────────────────────────────────────
        self._gmc = GlobalMotionCompensator()

        # ── Debug / HUD state ──────────────────────────────────────────────────
        self.flow_pts_display : Optional[np.ndarray] = None
        self.flow_health      : float = 0.0
        self.match_ref_source : str   = ""
        self.last_match_jump  : float = 0.0
        self.debug_status     : str   = ""

        # ── Approach window ────────────────────────────────────────────────────
        self._APPROACH_WINDOW = 6
        self._diag_hist : deque = deque(maxlen=self._APPROACH_WINDOW)

        # ── Miss counter ───────────────────────────────────────────────────────
        self._det_miss_count : int = 0
        self._MAX_DET_MISSES : int = 6

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def flow_healthy(self) -> bool:
        return self._of_fail_streak < OF_FAIL_LIMIT

    @property
    def flow_strong(self) -> bool:
        return self.flow_healthy and self._seed_count >= FLOW_STRONG_MIN_SEEDS

    def update(self, detections: np.ndarray,
               frame: np.ndarray,
               det_ran: bool = False) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._gmc.update(gray, self.last_bbox)

        if self.primary_id is None:
            if det_ran and len(detections) > 0:
                self._lock(detections, gray)
                self._rebuild_all_tracks()
                self._prev_gray = gray
                return True
            self._prev_gray = gray
            self._update_health()
            return False

        of_ok = self._run_optical_flow(gray)

        if det_ran:
            if len(detections) > 0:
                best = self._match_detection(detections)
                if best is not None:
                    self._update_confirmed(best, gray)
                    self._det_miss_count = 0
                    self._rebuild_all_tracks()
                    self._prev_gray = gray
                    self._update_health()
                    return True

            self._det_miss_count += 1
            self.frames_since_det += 1
            self.debug_status = (f"MISS #{self._det_miss_count}  "
                                 f"flow={'OK' if self.flow_healthy else 'DEAD'}")
            print(f"[MISS]  det_dets={len(detections)}  "
                  f"consecutive={self._det_miss_count}  "
                  f"flow={'healthy' if self.flow_healthy else 'DEAD'}")

            if self._det_miss_count >= self._MAX_DET_MISSES:
                print(f"[LOST]  {self._det_miss_count} det misses -> unlocking")
                self.unlock()
                self._prev_gray = gray
                return False

        # Between detections: use flow if healthy and on-target
        if of_ok and self._flow_bbox is not None and self.flow_healthy and not self._flow_on_bg:
            self._update_from_flow()
            quality = "STRONG" if self.flow_strong else "WEAK"
            self.debug_status = (f"FLOW-{quality}  pts={self._of_pts_alive}  "
                                 f"seed={self._seed_count}  health={self.flow_health:.0%}")
        else:
            # Apply GMC to coast position so it follows camera
            if self._gmc.valid and self.last_bbox is not None:
                compensated = self._gmc.compensate_bbox(self.last_bbox)
                self.last_bbox = tuple(int(v) for v in compensated)
            bg_note = "  [ON-BG]" if self._flow_on_bg else ""
            self.debug_status = (f"COAST  streak={self._of_fail_streak}"
                                 f"  gmc={'OK' if self._gmc.valid else 'NO'}{bg_note}")

        self._rebuild_all_tracks()
        self._prev_gray = gray
        self._update_health()
        return self.is_locked

    def unlock(self):
        print(f"[UNLOCK] Released track #{self.primary_id}")
        self.primary_id       = None
        self.last_bbox        = None
        self.frames_since_det = 0
        self._last_conf       = 0.0
        self._flow_pts        = None
        self._flow_bbox       = None
        self._of_box_hw       = None
        self._of_fail_streak  = 0
        self._of_pts_alive    = 0
        self._seed_count      = 0
        self._flow_on_bg      = False
        self.velocity         = None
        self.approach_str     = ""
        self.all_tracks       = []
        self.trail.clear()
        self._diag_hist.clear()
        self._smooth_state    = [0.0, 0.0]
        self.smooth_error     = (0.0, 0.0)
        self.raw_error        = (0.0, 0.0)
        self._det_miss_count  = 0
        self.flow_pts_display = None
        self.flow_health      = 0.0
        self.match_ref_source = ""
        self.debug_status     = "UNLOCKED"
        self._gmc.reset()

    @property
    def is_locked(self) -> bool:
        return self.primary_id is not None

    @property
    def is_coasting(self) -> bool:
        return self.is_locked and self.frames_since_det > 0

    @property
    def is_lost(self) -> bool:
        return self.primary_id is None

    # =========================================================================
    # Private
    # =========================================================================

    def _update_health(self):
        self.flow_health = max(0.0, 1.0 - self._of_fail_streak / OF_FAIL_LIMIT)

    def _lock(self, detections: np.ndarray, gray: np.ndarray):
        best = max(detections, key=lambda d: float(d[4]))
        self.primary_id       = 1
        self._det_miss_count  = 0
        self._of_fail_streak  = 0
        self.frames_since_det = 0
        print(f"[LOCK]  conf={best[4]:.2f}  "
              f"bbox=({int(best[0])},{int(best[1])},{int(best[2])},{int(best[3])})")
        self._seed_flow(best, gray)
        self._update_confirmed(best, gray)

    def _seed_flow(self, det: np.ndarray, gray: np.ndarray):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        
        # Dynamic margin: Shrink by 25% on all sides to avoid capturing background
        w = x2 - x1
        h = y2 - y1
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)

        x1m = max(x1 + margin_x, 0)
        y1m = max(y1 + margin_y, 0)
        x2m = min(x2 - margin_x, gray.shape[1] - 1)
        y2m = min(y2 - margin_y, gray.shape[0] - 1)

        if x2m <= x1m or y2m <= y1m:
            self._flow_pts   = None
            self._seed_count = 0
            return

        roi = gray[y1m:y2m, x1m:x2m]
        pts = cv2.goodFeaturesToTrack(roi, **FEATURE_PARAMS)

        if pts is None or len(pts) == 0:
            xs = np.linspace(x1m + 2, x2m - 2, 5)
            ys = np.linspace(y1m + 2, y2m - 2, 5)
            pts = np.array([[[x, y]] for y in ys for x in xs], dtype=np.float32)
        else:
            pts[:, :, 0] += x1m
            pts[:, :, 1] += y1m

        self._flow_pts       = pts.astype(np.float32)
        self._of_box_hw      = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)
        self._of_fail_streak = 0
        self._seed_count     = len(self._flow_pts)
        self._flow_on_bg     = False
        quality = "STRONG" if self._seed_count >= FLOW_STRONG_MIN_SEEDS else "WEAK"
        print(f"[SEED]  {self._seed_count} pts  bbox=({x1},{y1},{x2},{y2})  [{quality}]")

    def _run_optical_flow(self, gray: np.ndarray) -> bool:
        if self._prev_gray is None or self._flow_pts is None or len(self._flow_pts) == 0:
            return False

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._flow_pts, None, **LK_PARAMS
        )

        if new_pts is None or status is None:
            self._of_fail_streak += 1
            self._of_pts_alive = 0
            return False

        mask     = status.ravel() == 1
        good_new = new_pts[mask].reshape(-1, 2)
        good_old = self._flow_pts[mask].reshape(-1, 2)
        survived = len(good_new)
        self._of_pts_alive = survived

        if survived < MIN_FLOW_POINTS:
            self._of_fail_streak += 1
            if self._of_fail_streak >= OF_FAIL_LIMIT:
                self._flow_bbox  = None
                self._flow_on_bg = False
                print(f"[OF]    DEAD -- fail_streak={self._of_fail_streak} "
                      f"-> _flow_bbox cleared, will use last_bbox as match ref")
            else:
                print(f"[OF]    {survived} pts survived  "
                      f"fail_streak={self._of_fail_streak}/{OF_FAIL_LIMIT}")
            return False

        self._of_fail_streak = 0

        disp = good_new - good_old
        vx   = float(np.median(disp[:, 0]))
        vy   = float(np.median(disp[:, 1]))
        self.velocity = (vx, vy)

        # ── Flow-on-background detection (FIXED) ──────────────────────────────
        # Only check when the camera is actually moving meaningfully.
        # Compare flow VECTOR to camera motion VECTOR per-component.
        # When cam_disp < MIN_CAM_DISP, skip — can't distinguish target
        # from background when nothing is moving.
        if self._gmc.valid and self._gmc.cam_disp >= MIN_CAM_DISP:
            dx_err = abs(vx - self._gmc.cam_tx)
            dy_err = abs(vy - self._gmc.cam_ty)
            self._flow_on_bg = (dx_err < FLOW_BG_VEC_THRESH_X and
                                dy_err < FLOW_BG_VEC_THRESH_Y)
            if self._flow_on_bg:
                print(f"[OF]    ON-BG  flow=({vx:.1f},{vy:.1f})  "
                      f"cam=({self._gmc.cam_tx:.1f},{self._gmc.cam_ty:.1f})  "
                      f"err=({dx_err:.1f},{dy_err:.1f})")
        else:
            self._flow_on_bg = False

        # Reconstruct bbox from centroid
        cx = float(np.median(good_new[:, 0]))
        cy = float(np.median(good_new[:, 1]))
        hw, hh = self._of_box_hw if self._of_box_hw else (30.0, 30.0)
        x1 = int(np.clip(cx - hw, 0, self.frame_w))
        y1 = int(np.clip(cy - hh, 0, self.frame_h))
        x2 = int(np.clip(cx + hw, 0, self.frame_w))
        y2 = int(np.clip(cy + hh, 0, self.frame_h))
        self._flow_bbox = (x1, y1, x2, y2)

        self._flow_pts        = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.flow_pts_display = good_new
        return True

    def _match_detection(self, detections: np.ndarray) -> Optional[np.ndarray]:
        """
        Reference priority:
          1. FLOW-STRONG — healthy, >=10 seeds, not on background
          2. FLOW-WEAK   — healthy but sparse or on background
          3. GMC         — flow dead, GMC-compensated last_bbox
          4. DET         — flow dead, no GMC, raw last_bbox
          5. CONF        — no reference at all
        """
        use_flow = (self.flow_healthy
                    and self._flow_bbox is not None
                    and not self._flow_on_bg)

        if use_flow:
            ref = self._flow_bbox
            if self.flow_strong:
                max_jump = MAX_JUMP_FLOW_STRONG
                self.match_ref_source = "FLOW-STRONG"
            else:
                max_jump = MAX_JUMP_FLOW_WEAK
                self.match_ref_source = "FLOW-WEAK"

        elif self.last_bbox is not None:
            if self._gmc.valid:
                compensated = self._gmc.compensate_bbox(self.last_bbox)
                ref      = compensated
                max_jump = MAX_JUMP_GMC
                self.match_ref_source = "GMC"
                print(f"[GMC]   compensated ref: "
                      f"({self.last_bbox[0]:.0f},{self.last_bbox[1]:.0f},"
                      f"{self.last_bbox[2]:.0f},{self.last_bbox[3]:.0f}) -> "
                      f"({compensated[0]:.0f},{compensated[1]:.0f},"
                      f"{compensated[2]:.0f},{compensated[3]:.0f})")
            else:
                ref      = self.last_bbox
                max_jump = MAX_JUMP_FLOW_DEAD
                self.match_ref_source = "DET"
        else:
            self.match_ref_source = "CONF"
            return max(detections, key=lambda d: float(d[4]))

        rx    = (ref[0] + ref[2]) / 2.0
        ry    = (ref[1] + ref[3]) / 2.0
        max_r = np.hypot(self.frame_w, self.frame_h) * 0.45

        best_score, best_det = -float("inf"), None
        for d in detections:
            cx   = (d[0] + d[2]) / 2.0
            cy   = (d[1] + d[3]) / 2.0
            dist = np.hypot(cx - rx, cy - ry)
            conf = float(d[4])

            if dist > max_r:
                continue

            # Low-conf + large-jump gate
            if conf < LOW_CONF_THRESH and dist > LOW_CONF_MAX_JUMP:
                continue

            norm_prox = 1.0 - dist / max_r
            score     = 0.4 * conf + 0.6 * norm_prox

            if score > best_score:
                best_score, best_det = score, d

        if best_det is not None:
            bx   = int((best_det[0] + best_det[2]) / 2)
            by   = int((best_det[1] + best_det[3]) / 2)
            jump = float(np.hypot(bx - rx, by - ry))
            self.last_match_jump = jump

            if jump > max_jump:
                print(f"[MATCH] REJECTED  jump={jump:.0f}px > {max_jump}px  "
                      f"ref_src={self.match_ref_source}  "
                      f"ref=({rx:.0f},{ry:.0f})  cand=({bx},{by})")
                return None

            print(f"[DET]   matched  conf={best_det[4]:.2f}  "
                  f"ref_src={self.match_ref_source}  "
                  f"ref=({rx:.0f},{ry:.0f})  hit=({bx},{by})  jump={jump:.0f}px")
        return best_det

    def _update_confirmed(self, det: np.ndarray, gray: np.ndarray):
        x1   = int(det[0]); y1 = int(det[1])
        x2   = int(det[2]); y2 = int(det[3])
        conf = float(det[4])
        tcx  = (x1 + x2) // 2
        tcy  = (y1 + y2) // 2
        diag = float(np.hypot(x2 - x1, y2 - y1))

        self._diag_hist.append(diag)
        if len(self._diag_hist) == self._APPROACH_WINDOW:
            hist  = list(self._diag_hist)
            half  = self._APPROACH_WINDOW // 2
            delta = np.mean(hist[half:]) - np.mean(hist[:half])
            self.approach_str = ("CLOSING"  if delta >  2.0
                            else "RECEDING" if delta < -2.0 else "")

        self.last_bbox        = (x1, y1, x2, y2)
        self._last_conf       = conf
        self.frames_since_det = 0
        self.trail.append((tcx, tcy))
        self._update_error(tcx, tcy)
        self.debug_status = (f"DET  conf={conf:.2f}  jump={self.last_match_jump:.0f}px  "
                             f"ref={self.match_ref_source}")
        self._seed_flow(det, gray)

    def _update_from_flow(self):
        if self._flow_bbox is None:
            return
        x1, y1, x2, y2 = self._flow_bbox
        tcx = (x1 + x2) // 2
        tcy = (y1 + y2) // 2
        self.trail.append((tcx, tcy))
        self._update_error(tcx, tcy)

    def _update_error(self, tcx: int, tcy: int):
        ex = float(np.clip((tcx - self._cx) / (self.frame_w / 2.0), -1.0, 1.0))
        ey = float(np.clip((tcy - self._cy) / (self.frame_h / 2.0), -1.0, 1.0))
        self.raw_error = (ex, ey)
        a = config.ERROR_SMOOTH_ALPHA
        self._smooth_state[0] = a * ex + (1 - a) * self._smooth_state[0]
        self._smooth_state[1] = a * ey + (1 - a) * self._smooth_state[1]
        self.smooth_error = (self._smooth_state[0], self._smooth_state[1])

    def _rebuild_all_tracks(self):
        if not self.is_locked:
            self.all_tracks = []
            return
        if self.flow_healthy and self._flow_bbox is not None and not self._flow_on_bg:
            bbox = self._flow_bbox
        elif self.last_bbox is not None:
            bbox = self.last_bbox
        else:
            self.all_tracks = []
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        row = np.array([x1, y1, x2, y2, float(self.primary_id), self._last_conf],
                       dtype=np.float32)
        self.all_tracks = [row]