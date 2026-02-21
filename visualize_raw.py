# =============================================================================
# visualize_raw.py — Raw Detector Visualization (No Tracker)
#
# Usage:
#   python visualize_raw.py
#
# Purpose:
#   See exactly what the detector outputs and how the "Skipped Frames"
#   look visually.
#   - GREEN Box = Fresh Detection (Model ran this frame)
#   - RED Box   = Stale/Held Detection (Model skipped this frame)
# =============================================================================

import cv2
import numpy as np
import time
import sys

import config
from detector import build_detector

def main():
    # 1. Setup
    print(f"Loading Detector: {config.DETECTOR}...")
    detector = build_detector()
    every_n  = config.detector_every_n()
    
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {config.VIDEO_PATH}")
        return

    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Output video
    out_path = config.OUTPUT_PATH.replace(".mp4", "_raw_debug.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    print(f"\n── Raw Detector Test ──────────────────────────────────")
    print(f"Detector : {detector.name}")
    print(f"Interval : Run every {every_n} frames")
    print(f"Video    : {frame_w}x{frame_h} @ {fps:.1f} FPS")
    print(f"Saving to: {out_path}")
    print("───────────────────────────────────────────────────────\n")

    frame_idx = 0
    last_detections = np.empty((0, 6))
    last_det_time = 0.0
    
    # Pause on first frame?
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        display_frame = frame.copy()
        
        # 2. Run Detector
        is_det_frame = (frame_idx % every_n == 0)
        det_start = time.perf_counter()
        
        if is_det_frame:
            # Run inference
            t0 = time.perf_counter()
            detections = detector(frame)
            t1 = time.perf_counter()
            
            # Store for visualization
            last_detections = detections
            last_det_time = (t1 - t0) * 1000.0 # ms
            
            box_color = (0, 255, 0) # Green for FRESH
            status_text = f"FRESH DET ({last_det_time:.0f}ms)"
        else:
            # holds last detection
            box_color = (0, 0, 255) # Red for STALE
            status_text = f"STALE (+{frame_idx % every_n})"

        # 3. Draw Boxes
        if len(last_detections) > 0:
            for i, det in enumerate(last_detections):
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw Box
                thickness = 2 if is_det_frame else 1
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # Draw Label
                label = f"{detector.name}: {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + t_size[0], y1), box_color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4. HUD Info
        hud_color = (0, 255, 0) if is_det_frame else (0, 0, 255)
        
        # Top bar background
        cv2.rectangle(display_frame, (0, 0), (frame_w, 40), (0,0,0), -1)
        
        info = f"Frame: {frame_idx}/{total_frames} | {status_text} | Dets: {len(last_detections)}"
        cv2.putText(display_frame, info, (10, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

        if not is_det_frame:
            cv2.putText(display_frame, "(Holding last known position)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 5. Display & Control
        cv2.imshow("Raw Detector Debug", display_frame)
        writer.write(display_frame)

        if not paused:
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): # Space to pause/resume
            paused = not paused

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()