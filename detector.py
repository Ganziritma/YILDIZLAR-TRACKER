# =============================================================================
# detector.py — Unified detector interface
# Supports: YOLO (ultralytics) and RF-DETR Medium (your checkpoint)
# Both return np.ndarray (N, 6) → [x1, y1, x2, y2, conf, cls]
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import cv2
from abc import ABC, abstractmethod
from PIL import Image

import config


class BaseDetector(ABC):
    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── YOLO ──────────────────────────────────────────────────────────────────────

class YOLODetector(BaseDetector):

    def __init__(self):
        from ultralytics import YOLO
        self._model = YOLO(config.MODEL_PATH)
        print(f"[DETECTOR] YOLO loaded from {config.MODEL_PATH}")

    @property
    def name(self) -> str:
        return "YOLO"

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        results = self._model(
            frame, verbose=False,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
        )[0]
        boxes = results.boxes
        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        raw = np.column_stack([
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ]).astype(np.float32)
        return _filter_large(raw, frame.shape)


# ── RF-DETR ───────────────────────────────────────────────────────────────────

class RFDETRDetector(BaseDetector):
    """
    RF-DETR Medium wrapper.

    NOTE: RF-DETR with DINOv2 backbone is very heavy.
    Expected speed:
      CPU only (e.g. Aspire 3) : ~1-3 FPS inference → set TRANSFORMER_EVERY_N=30
      GPU (RTX 3060+)          : ~15-30 FPS inference → TRANSFORMER_EVERY_N=3
    """

    def __init__(self):
        from rfdetr import RFDETRMedium

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = config.TRANSFORMER_CONF
        self.imgsz     = config.TRANSFORMER_IMGSZ

        print(f"[DETECTOR] Loading RF-DETR Medium from {config.TRANSFORMER_PATH} "
              f"on {self.device}")
        if self.device.type == "cpu":
            print("[DETECTOR] WARNING: RF-DETR on CPU is very slow (~1-3 FPS). "
                  "Consider switching to DETECTOR='yolo' in config.py for real-time use.")

        # pretrain_weights accepts a local .pth path — RF-DETR loads it internally
        self._rfdetr = RFDETRMedium(
            num_classes=config.TRANSFORMER_NUM_CLASSES,
            resolution=config.TRANSFORMER_IMGSZ,
            pretrain_weights=config.TRANSFORMER_PATH,
        )

        # Optimise the model graph for inference (removes training-only ops)
        try:
            self._rfdetr.model.optimize_for_inference()
            print("[DETECTOR] Model optimised for inference.")
        except Exception as e:
            print(f"[DETECTOR] Could not optimise for inference: {e}")

        print(f"[DETECTOR] RF-DETR ready on {self.device} "
              f"(resolution={self.imgsz}, threshold={self.threshold})")

    @property
    def name(self) -> str:
        return "RF-DETR"

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result  = self._rfdetr.predict(pil_img, threshold=self.threshold)

        if result is None or len(result.xyxy) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes  = np.array(result.xyxy,       dtype=np.float32)
        confs  = np.array(result.confidence, dtype=np.float32).reshape(-1, 1)
        labels = np.array(result.class_id,   dtype=np.float32).reshape(-1, 1)

        return _filter_large(np.hstack([boxes, confs, labels]), frame.shape)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _filter_large(raw: np.ndarray, shape: tuple) -> np.ndarray:
    if len(raw) == 0:
        return np.empty((0, 6), dtype=np.float32)
    h, w  = shape[:2]
    areas = (raw[:, 2] - raw[:, 0]) * (raw[:, 3] - raw[:, 1])
    valid = areas < (config.YOLO_MAX_AREA_RATIO * w * h)
    return raw[valid].astype(np.float32) if valid.any() else np.empty((0, 6), dtype=np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_detector() -> BaseDetector:
    if config.DETECTOR == "transformer":
        return RFDETRDetector()
    elif config.DETECTOR == "yolo":
        return YOLODetector()
    else:
        raise ValueError(f"Unknown config.DETECTOR: {config.DETECTOR!r}. "
                         "Choose 'yolo' or 'transformer'.")
