# =============================================================================
# deep_benchmark.py — RF-DETR + Optical Flow Tracker Diagnostics
#
# Runs through the video and measures everything that matters for tracker
# health, then prints a full report with concrete config recommendations.
#
# Usage:
#   python deep_benchmark.py
#   python deep_benchmark.py --frames 300   # only first N frames
#   python deep_benchmark.py --no-video     # skip debug video output
#
# Output:
#   1. Console table  — per-detection-frame stats
#   2. Console report — summary + tuning recommendations
#   3. *_benchmark.mp4 — annotated debug video (unless --no-video)
# =============================================================================

import cv2
import numpy as np
import time
import argparse
from collections import deque

import config
from detector import build_detector

parser = argparse.ArgumentParser()
parser.add_argument("--frames",   type=int, default=999999)
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()

# ── LK params (same as tracker_core so benchmark reflects reality) ────────────
LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEATURE_PARAMS = dict(
    maxCorners=50, qualityLevel=0.005, minDistance=3, blockSize=3,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
print("Loading detector …")
detector = build_detector()
every_n  = config.detector_every_n()

cap      = cv2.VideoCapture(config.VIDEO_PATH)
frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
diag_px  = np.hypot(frame_w, frame_h)

writer = None
if not args.no_video:
    out_path = config.OUTPUT_PATH.replace(".mp4", "_benchmark.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    panel_h  = 200
    writer   = cv2.VideoWriter(out_path, fourcc, fps_vid,
                               (frame_w, frame_h + panel_h))
    print(f"[VIDEO] → {out_path}")

print(f"Video   : {frame_w}×{frame_h} @ {fps_vid:.1f}fps  ({total_f} frames)")
print(f"Detector: every {every_n} frames\n")

# ── Per-frame accumulators ────────────────────────────────────────────────────
inf_times_ms   = []          # inference latency per detection call
n_dets_list    = []          # number of detections per detection frame
conf_all       = []          # every detection confidence seen
bbox_areas     = []          # every detection bbox area (px²)
bbox_widths    = []
bbox_heights   = []
of_survived    = []          # % flow points surviving each non-det frame
of_seeded      = []          # points seeded per detection
match_jumps    = []          # px distance between consecutive confirmed bboxes
det_frame_idxs = []          # which frames were detection frames

# ── State for OF simulation ───────────────────────────────────────────────────
prev_gray    = None
flow_pts     = None
of_box_hw    = None
last_bbox    = None
last_cx      = None
last_cy      = None

def seed_flow(gray, x1, y1, x2, y2):
    margin = 3
    x1m = max(x1+margin, 0); y1m = max(y1+margin, 0)
    x2m = min(x2-margin, gray.shape[1]-1)
    y2m = min(y2-margin, gray.shape[0]-1)
    if x2m <= x1m or y2m <= y1m:
        return None, None
    roi = gray[y1m:y2m, x1m:x2m]
    pts = cv2.goodFeaturesToTrack(roi, **FEATURE_PARAMS)
    if pts is None or len(pts) == 0:
        xs = np.linspace(x1m+2, x2m-2, 5)
        ys = np.linspace(y1m+2, y2m-2, 5)
        pts = np.array([[[x, y]] for y in ys for x in xs], dtype=np.float32)
    else:
        pts[:,:,0] += x1m
        pts[:,:,1] += y1m
    hw = (x2 - x1) / 2.0
    hh = (y2 - y1) / 2.0
    return pts.astype(np.float32), (hw, hh)

# ── Column header ─────────────────────────────────────────────────────────────
HDR = (f"{'F':>5} {'INF_MS':>7} {'N_DET':>5} {'CONF_MAX':>8} "
       f"{'BOX_WxH':>12} {'BOX_AREA':>8} {'OF_PTS':>6} {'OF_SURV%':>8} {'JUMP_PX':>7}")
print(HDR)
print("─" * len(HDR))

# ── Main loop ─────────────────────────────────────────────────────────────────
frame_idx  = 0
last_dets  = np.empty((0,6), dtype=np.float32)

while frame_idx < min(args.frames, total_f):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_det = (frame_idx % every_n == 0)

    of_surv_pct = None
    inf_ms      = None
    n_det       = 0
    conf_max    = None
    bw = bh     = None
    seeded_n    = None
    jump_px     = None
    dets        = np.empty((0,6), dtype=np.float32)

    # ── Optical flow (every frame) ─────────────────────────────────────────
    if prev_gray is not None and flow_pts is not None and len(flow_pts) > 0:
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, flow_pts, None, **LK_PARAMS)
        if new_pts is not None and status is not None:
            good = new_pts[status.ravel() == 1].reshape(-1, 2)  # (N,1,2) → (N,2)
            total_pts = len(flow_pts)
            surv = len(good)
            of_surv_pct = 100.0 * surv / max(total_pts, 1)
            of_survived.append(of_surv_pct)
            if surv >= 2:
                flow_pts = good.reshape(-1, 1, 2).astype(np.float32)
            # Update flow bbox centroid
            if surv >= 2 and of_box_hw is not None:
                cx = float(np.median(good[:, 0]))
                cy = float(np.median(good[:, 1]))
                hw, hh = of_box_hw
                last_bbox = (int(cx-hw), int(cy-hh), int(cx+hw), int(cy+hh))

    # ── Detector ───────────────────────────────────────────────────────────
    if is_det:
        t0     = time.perf_counter()
        dets   = detector(frame)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        inf_times_ms.append(inf_ms)
        det_frame_idxs.append(frame_idx)
        n_det  = len(dets)
        n_dets_list.append(n_det)

        if n_det > 0:
            for d in dets:
                conf_all.append(float(d[4]))
                w = float(d[2]-d[0]); h = float(d[3]-d[1])
                bbox_areas.append(w*h)
                bbox_widths.append(w); bbox_heights.append(h)
            # Best detection by confidence
            best   = max(dets, key=lambda d: float(d[4]))
            conf_max = float(best[4])
            bw     = float(best[2]-best[0]); bh = float(best[3]-best[1])
            bx1,by1,bx2,by2 = int(best[0]),int(best[1]),int(best[2]),int(best[3])

            # Jump distance from previous confirmed
            new_cx = (bx1+bx2)/2; new_cy = (by1+by2)/2
            if last_cx is not None:
                jump_px = float(np.hypot(new_cx-last_cx, new_cy-last_cy))
                match_jumps.append(jump_px)
            last_cx, last_cy = new_cx, new_cy

            # Re-seed OF from detection
            pts, hw_hh = seed_flow(gray, bx1, by1, bx2, by2)
            if pts is not None:
                flow_pts  = pts
                of_box_hw = hw_hh
                seeded_n  = len(pts)
                of_seeded.append(seeded_n)
            last_bbox = (bx1, by1, bx2, by2)

    # ── Console row ────────────────────────────────────────────────────────
    if is_det:
        surv_str  = f"{of_surv_pct:>7.1f}%" if of_surv_pct is not None else "     N/A"
        conf_str  = f"{conf_max:.2f}" if conf_max else "  —"
        bwh_str   = f"{bw:.0f}×{bh:.0f}" if bw else "  —"
        area_str  = f"{bw*bh:.0f}" if bw else "   —"
        jump_str  = f"{jump_px:.0f}" if jump_px is not None else "  —"
        seed_str  = f"{seeded_n}" if seeded_n else " —"
        print(f"{frame_idx:>5} {inf_ms:>7.0f} {n_det:>5} {conf_str:>8} "
              f"{bwh_str:>12} {area_str:>8} {seed_str:>6} {surv_str:>8} {jump_str:>7}")

    # ── Video annotation ───────────────────────────────────────────────────
    if writer is not None:
        vis = frame.copy()
        # Draw all detections
        for d in dets:
            x1,y1,x2,y2 = int(d[0]),int(d[1]),int(d[2]),int(d[3])
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,230,0),1)
            cv2.putText(vis,f"{d[4]:.2f}",(x1,max(y1-3,10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,230,0),1)
        # Draw flow points
        if flow_pts is not None:
            for pt in flow_pts.reshape(-1,2):
                cv2.circle(vis,(int(pt[0]),int(pt[1])),2,(0,255,255),-1)
        # Draw tracked bbox
        if last_bbox:
            x1,y1,x2,y2 = last_bbox
            cv2.rectangle(vis,(x1,y1),(x2,y2),(255,100,0),2)

        # Info panel
        panel = np.zeros((200, frame_w, 3), np.uint8)
        def pp(txt, row, col=(180,180,180)):
            cv2.putText(panel, txt, (8, 20+row*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        pp(f"Frame {frame_idx}  |  det_frame={'YES' if is_det else 'no'}  "
           f"|  inf={inf_ms:.0f}ms" if inf_ms else
           f"Frame {frame_idx}  |  det_frame=no", 0)
        pp(f"Detections: {n_det}   max_conf={conf_max:.2f}" if conf_max else
           f"Detections: {n_det}", 1, (0,230,0) if n_det else (80,80,80))
        if bw:
            pp(f"Best bbox: {bw:.0f}×{bh:.0f}px  area={bw*bh:.0f}px²  "
               f"(small if area<3000)", 2,
               (0,80,255) if bw*bh < 3000 else (0,230,0))
        seeded_str = f"{seeded_n}" if seeded_n else "—"
        pp(f"OF points seeded: {seeded_str}   "
           f"surviving: {of_surv_pct:.1f}%" if of_surv_pct is not None else
           f"OF points seeded: {seeded_str}", 3,
           (0,80,255) if (of_surv_pct or 100) < 30 else (180,180,180))
        if jump_px is not None:
            pp(f"Consecutive det jump: {jump_px:.0f}px  "
               f"({'OK' if jump_px<120 else 'SUSPICIOUS'})", 4,
               (0,230,0) if jump_px < 120 else (0,80,255))

        writer.write(np.vstack([vis, panel]))

    prev_gray = gray
    frame_idx += 1

cap.release()
if writer:
    writer.release()

# =============================================================================
# REPORT
# =============================================================================
print("\n" + "═"*65)
print("  BENCHMARK REPORT")
print("═"*65)

# ── Inference speed ───────────────────────────────────────────────────────────
if inf_times_ms:
    med_ms = np.median(inf_times_ms)
    p90_ms = np.percentile(inf_times_ms, 90)
    inf_fps = 1000.0 / med_ms
    # How many video frames pass during one inference?
    frames_per_inf = med_ms / (1000.0 / fps_vid)
    # Recommended every_n: no point submitting faster than inference completes
    rec_every_n = max(1, int(np.ceil(frames_per_inf)))

    print(f"\n── INFERENCE SPEED ──────────────────────────────────────")
    print(f"  Median latency : {med_ms:.0f} ms")
    print(f"  P90 latency    : {p90_ms:.0f} ms")
    print(f"  Effective FPS  : {inf_fps:.2f} FPS")
    print(f"  Frames per inf : {frames_per_inf:.1f}  (at {fps_vid:.1f}fps video)")
    print(f"\n  Current  TRANSFORMER_EVERY_N = {every_n}")
    print(f"  Optimal  TRANSFORMER_EVERY_N = {rec_every_n}  "
          f"({'same' if rec_every_n==every_n else '← change this'})")
    if rec_every_n < every_n:
        print(f"  → You are waiting longer than needed. "
              f"Set EVERY_N={rec_every_n} to get detections as fast as possible.")
    elif rec_every_n > every_n:
        print(f"  → You are submitting faster than inference finishes. "
              f"The async thread ignores surplus submissions (no harm, no gain). "
              f"Set EVERY_N={rec_every_n} for clarity.")

# ── Detection health ──────────────────────────────────────────────────────────
if n_dets_list:
    miss_rate = 100.0 * sum(1 for n in n_dets_list if n==0) / len(n_dets_list)
    multi_rate= 100.0 * sum(1 for n in n_dets_list if n>1)  / len(n_dets_list)
    print(f"\n── DETECTION HEALTH ─────────────────────────────────────")
    print(f"  Detection frames  : {len(n_dets_list)}")
    print(f"  Empty frames      : {miss_rate:.1f}%")
    print(f"  Multi-det frames  : {multi_rate:.1f}%")
    if miss_rate > 20:
        print(f"  ⚠  High miss rate — consider lowering TRANSFORMER_CONF "
              f"(currently {config.TRANSFORMER_CONF})")
    if miss_rate < 5 and multi_rate > 30:
        print(f"  ⚠  Many multi-det frames — consider raising TRANSFORMER_CONF "
              f"to reduce false positives")

if conf_all:
    print(f"\n── CONFIDENCE DISTRIBUTION ──────────────────────────────")
    print(f"  Min  : {np.min(conf_all):.2f}")
    print(f"  Mean : {np.mean(conf_all):.2f}")
    print(f"  P50  : {np.median(conf_all):.2f}")
    print(f"  P90  : {np.percentile(conf_all,90):.2f}")
    print(f"  Max  : {np.max(conf_all):.2f}")
    print(f"  Current TRANSFORMER_CONF = {config.TRANSFORMER_CONF}")
    sweet_spot = float(np.percentile(conf_all, 25))
    print(f"  Suggested TRANSFORMER_CONF ≈ {sweet_spot:.2f}  (25th-pct of seen confs)")

# ── Bounding box size ─────────────────────────────────────────────────────────
if bbox_areas:
    print(f"\n── BOUNDING BOX SIZE ────────────────────────────────────")
    print(f"  Median area  : {np.median(bbox_areas):.0f} px²")
    print(f"  Median W×H   : {np.median(bbox_widths):.0f}×{np.median(bbox_heights):.0f} px")
    print(f"  Min area     : {np.min(bbox_areas):.0f} px²")
    print(f"  Max area     : {np.max(bbox_areas):.0f} px²")
    med_area = np.median(bbox_areas)
    if med_area < 1500:
        print(f"  ⚠  Very small targets (median {med_area:.0f}px²).")
        print(f"     OF will find few corners. MIN_FLOW_POINTS=2 is correct.")
        print(f"     If tracking is still poor, try KCFN or MIL tracker from")
        print(f"     cv2.TrackerKCF_create() which handles small objects better.")
    elif med_area < 4000:
        print(f"  ℹ  Small-to-medium targets. Current OF params should be OK.")
    else:
        print(f"  ✓  Large enough targets for reliable optical flow.")

# ── Optical flow health ───────────────────────────────────────────────────────
if of_survived:
    mean_surv = np.mean(of_survived)
    low_surv  = sum(1 for s in of_survived if s < 30)
    print(f"\n── OPTICAL FLOW HEALTH ──────────────────────────────────")
    print(f"  Mean survival rate : {mean_surv:.1f}%")
    print(f"  Frames < 30% surv  : {low_surv} / {len(of_survived)}")
    if mean_surv < 40:
        print(f"  ⚠  Low OF survival. Likely causes:")
        print(f"     1. Target is very small (see bbox size above)")
        print(f"     2. Fast camera/target motion — try reducing LK winSize to (11,11)")
        print(f"     3. Low-texture target surface — OF struggles on smooth fuselages")
        print(f"     Recommendation: keep EVERY_N low so det re-anchors frequently")

if of_seeded:
    print(f"  Mean pts seeded    : {np.mean(of_seeded):.1f}")
    print(f"  Min pts seeded     : {np.min(of_seeded)}")
    if np.min(of_seeded) < 4:
        print(f"  ⚠  Some frames seed fewer than 4 points. The tracker will")
        print(f"     rely on the fallback grid in those cases.")

# ── Consecutive detection jumps ───────────────────────────────────────────────
if match_jumps:
    print(f"\n── DETECTION JUMP DISTANCES (frame-to-frame) ────────────")
    print(f"  Median jump  : {np.median(match_jumps):.0f} px")
    print(f"  P90 jump     : {np.percentile(match_jumps,90):.0f} px")
    print(f"  Max jump     : {np.max(match_jumps):.0f} px")
    big = sum(1 for j in match_jumps if j > 120)
    print(f"  Jumps > 120px: {big} / {len(match_jumps)} "
          f"({'these will be rejected as wrong matches' if big else '✓ none'})")
    p90_jump = np.percentile(match_jumps, 90)
    print(f"  Suggested MAX_MATCH_JUMP_PX ≈ {p90_jump*1.5:.0f}  "
          f"(1.5× the P90 jump — covers fast real motion with margin)")

# ── Final config recommendations ──────────────────────────────────────────────
print(f"\n── CONFIG RECOMMENDATIONS (config.py) ───────────────────")
if inf_times_ms:
    print(f"  TRANSFORMER_EVERY_N  = {rec_every_n}  "
          f"(was {every_n})")
if conf_all:
    print(f"  TRANSFORMER_CONF     = {sweet_spot:.2f}  "
          f"(was {config.TRANSFORMER_CONF})")
print(f"\n── tracker_core.py constants ────────────────────────────")
print(f"  MIN_FLOW_POINTS  = 2   (already set — correct for small targets)")
if match_jumps:
    rec_jump = int(np.percentile(match_jumps,90) * 1.5)
    print(f"  MAX_MATCH_JUMP_PX = {rec_jump}  (was 120)")
print()