"""YOLOv8-based object detection for per-frame player counting.

Runs YOLO inference on video frames (with configurable sampling rate) and
produces per-frame player counts and bounding box statistics.

The model has 4 classes: ball=0, goalkeeper=1, player=2, referee=3.
For people counting, all non-ball classes (1,2,3) are summed.

Note: A 2-class version (ball=0, players=1) was also trained by merging
goalkeeper/player/referee into a single class, but the 4-class model
significantly outperforms it in detection coverage.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

from .config import ObjectDetectionConfig
from .utils import get_video_properties

logger = logging.getLogger(__name__)


class PlayerDetector:
    """Detect and count players per frame using YOLOv8."""

    # Model classes: ball=0, goalkeeper=1, player=2, referee=3
    # For people counting: all non-ball classes
    PEOPLE_CLASS_IDS = {1, 2, 3}  # goalkeeper, player, referee

    def __init__(self, config: Optional[ObjectDetectionConfig] = None):
        self.config = config or ObjectDetectionConfig()
        self.model = None

    def _load_model(self):
        """Load YOLO model lazily."""
        if self.model is not None:
            return

        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", self.config.model_path)
        self.model = YOLO(self.config.model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        logger.info("Model loaded on %s", device)

    def run(self, video_path: str) -> pd.DataFrame:
        """Run detection on a video and return per-frame player counts.

        Processes every `frame_step` frames for speed.

        Returns DataFrame with columns:
            frame, time_seconds, player_count, avg_box_diagonal
        """
        self._load_model()

        props = get_video_properties(video_path)
        fps = props["fps"]
        total_frames = props["total_frames"]
        step = self.config.frame_step

        cap = cv2.VideoCapture(video_path)
        results_data = []
        conf_threshold = self.config.confidence_threshold

        logger.info(
            "Processing %d frames (every %d, effective %.1f fps)...",
            total_frames, step, fps / step,
        )

        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO on single frame
            preds = self.model.predict(frame, save=False, verbose=False, conf=conf_threshold)

            player_count = 0
            box_diagonals = []

            if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
                boxes = preds[0].boxes
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    if class_id in self.PEOPLE_CLASS_IDS and confidence >= conf_threshold:
                        player_count += 1
                        coords = box.xyxy[0].cpu().numpy()
                        w = coords[2] - coords[0]
                        h = coords[3] - coords[1]
                        box_diagonals.append(np.sqrt(w**2 + h**2))

            avg_diagonal = float(np.mean(box_diagonals)) if box_diagonals else 0.0

            results_data.append({
                "frame": frame_idx,
                "time_seconds": frame_idx / fps,
                "player_count": player_count,
                "avg_box_diagonal": avg_diagonal,
            })

            frame_idx += step

        cap.release()

        df = pd.DataFrame(results_data)
        logger.info(
            "Detection complete: %d sampled frames, avg %.1f players/frame",
            len(df), df["player_count"].mean(),
        )
        return df
