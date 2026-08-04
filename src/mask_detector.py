"""
Mask Detector Module
--------------------
High-level orchestrator class combining Face Detection, Neural Classification,
bounding box rendering, and metric counters.

Author: Abid Ali
"""

import numpy as np
import cv2
from src.face_detector import FaceDetector
from src.model_loader import ModelLoader
from src.utils import draw_bounding_box
from config import settings


class MaskDetector:
    """
    Real-time Face Mask Detection pipeline orchestrator.
    """

    def __init__(self, face_confidence: float = settings.FACE_CONFIDENCE_THRESHOLD):
        """
        Initialize Mask Detector pipeline.

        :param face_confidence: Confidence threshold for face detection.
        """
        self.face_detector = FaceDetector(confidence_threshold=face_confidence)
        self.model_loader = ModelLoader(model_path=str(settings.MODEL_PATH))

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single video frame to detect faces, classify mask usage,
        and render bounding box overlays.

        :param frame: Video frame numpy array (BGR).
        :return: Tuple containing:
                 - annotated_frame (np.ndarray)
                 - list of detection objects
                 - mask_count (int)
                 - nomask_count (int)
        """
        if frame is None or frame.size == 0:
            return frame, [], 0, 0

        h, w = frame.shape[:2]
        faces = self.face_detector.detect_faces(frame)

        results = []
        mask_count = 0
        nomask_count = 0

        annotated_frame = frame.copy()

        for (x, y, box_w, box_h, face_conf) in faces:
            # Expand face ROI by 5% margin for better context
            margin_x = int(box_w * 0.05)
            margin_y = int(box_h * 0.05)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + box_w + margin_x)
            y2 = min(h, y + box_h + margin_y)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            # Run classifier inference
            label, mask_conf, is_mask = self.model_loader.predict(face_roi)

            # Update counters
            if is_mask:
                mask_count += 1
            else:
                nomask_count += 1

            detection_info = {
                "box": (x, y, box_w, box_h),
                "label": label,
                "confidence": mask_conf,
                "is_mask": is_mask,
                "face_confidence": face_conf
            }
            results.append(detection_info)

            # Draw visual overlay on annotated frame
            annotated_frame = draw_bounding_box(
                annotated_frame,
                (x, y, box_w, box_h),
                label=label,
                confidence=mask_conf,
                is_mask=is_mask
            )

        return annotated_frame, results, mask_count, nomask_count
