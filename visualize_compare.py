# =============================================================================
# visualize_compare.py — RAW vs TRACKER (STABLE + SAFE RECORD)
# =============================================================================

import cv2
import numpy as np
import time
from pathlib import Path

import config
from detector import build_detector
from tracker_core import AirCombatTracker
import hud


def main():

    detector = build_detector()
    every_n = config.detector_every_n()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    target_frame_time = 1.0 / native_fps

    # -----------------------------------------------------------------
    # GUARANTEED SAFE OUTPUT (AVI + XVID)
    # -----------------------------------------------------------------
    out_path = Path(config.OUTPUT_PATH).with_suffix(".avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        native_fps,
        (frame_w * 2, frame_h)
    )

    if not writer.isOpened():
        raise RuntimeError("VideoWriter açılamadı. Codec problemi olabilir.")

    tracker = AirCombatTracker(frame_w, frame_h)

    print("Recording automatically...")
    print(f"Output: {out_path}")
    print("Q = quit\n")

    frame_idx = 0
    last_detections = np.empty((0, 6), dtype=np.float32)

    cv2.namedWindow("RAW vs TRACKER", cv2.WINDOW_NORMAL)

    try:
        while True:

            frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            is_det_frame = (frame_idx % every_n == 0)

            # ================= RAW =================

            raw_display = frame.copy()

            if is_det_frame:
                detections = detector(frame)
                last_detections = detections.copy()
                box_color = (0, 255, 0)
            else:
                box_color = (0, 0, 255)

            for det in last_detections:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(raw_display, (x1, y1), (x2, y2), box_color, 2)

            cv2.putText(raw_display, "RAW",
                        (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

            # ================= TRACKER =================

            tracked_display = frame.copy()

            dets_for_tracker = last_detections if is_det_frame else np.empty((0, 6), dtype=np.float32)

            tracker.update(dets_for_tracker, frame, det_ran=is_det_frame)

            hud.draw_other_tracks(tracked_display, tracker.all_tracks, tracker.primary_id)

            primary_row = None
            for t in tracker.all_tracks:
                if int(t[4]) == tracker.primary_id:
                    primary_row = t
                    break

            if primary_row is not None:
                hud.draw_primary_track(
                    tracked_display, primary_row,
                    tracker.raw_error, tracker.smooth_error,
                    tracker.approach_str, tracker.velocity,
                    frame_w // 2, frame_h // 2,
                    frame_w, frame_h
                )

            cv2.putText(tracked_display, "TRACKER",
                        (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

            # ================= MERGE =================

            combined = np.hstack([raw_display, tracked_display])

            cv2.imshow("RAW vs TRACKER", combined)

            # WRITE EVERY FRAME (automatic)
            writer.write(combined)

            # ================= STABLE FPS =================

            elapsed = time.perf_counter() - frame_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Video saved successfully.")


if __name__ == "__main__":
    main()