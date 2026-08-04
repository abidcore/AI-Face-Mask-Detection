"""
Face Detector Module
--------------------
Handles multi-face detection using MediaPipe Face Detection API with fallback
to OpenCV Haar Cascade or Caffe Deep Neural Network detector.

Author: Abid Ali
"""

import os
import logging
import numpy as np
import cv2
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Multi-face detector supporting MediaPipe Face Detection and OpenCV Haar Cascade.
    """

    def __init__(self, confidence_threshold: float = settings.FACE_CONFIDENCE_THRESHOLD):
        """
        Initialize Face Detector.

        :param confidence_threshold: Minimum confidence score for valid face detection.
        """
        self.confidence_threshold = confidence_threshold
        self.mp_face_detector = None
        self.haar_cascade = None
        self.use_mediapipe = False

        self._init_mediapipe()
        if not self.use_mediapipe:
            self._init_haar_cascade()

    def _init_mediapipe(self) -> bool:
        """Attempt to initialize MediaPipe Face Detection solution."""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.confidence_threshold,
                model_selection=0  # Short-range model suitable for webcam/selfie
            )
            self.use_mediapipe = True
            logger.info("MediaPipe Face Detection engine initialized.")
        except Exception as e:
            logger.warning("MediaPipe initialization failed: %s. Switching to OpenCV Haar Cascade.", str(e))
            self.use_mediapipe = False
        return self.use_mediapipe

    def _init_haar_cascade(self) -> None:
        """Initialize OpenCV Haar Cascade Face Classifier."""
        cascade_path = settings.CASCADE_PATH
        if not os.path.exists(cascade_path):
            # Attempt to use built-in OpenCV cascade file
            default_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(default_xml):
                cascade_path = default_xml

        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(str(cascade_path))
            logger.info("OpenCV Haar Cascade initialized from: %s", cascade_path)
        else:
            logger.error("Haar Cascade XML model file not found!")

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in the input image frame.

        :param frame: Video frame numpy array (BGR).
        :return: List of tuples: [(x, y, w, h, confidence_score), ...]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections = []

        # Try MediaPipe detection
        if self.use_mediapipe and self.mp_face_detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detector.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        score = detection.score[0] if detection.score else 0.90
                        if score >= self.confidence_threshold:
                            bbox = detection.location_data.relative_bounding_box
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            box_w = int(bbox.width * w)
                            box_h = int(bbox.height * h)

                            # Clip coordinates within image boundaries
                            x = max(0, x)
                            y = max(0, y)
                            box_w = min(w - x, box_w)
                            box_h = min(h - y, box_h)

                            if box_w > 15 and box_h > 15:
                                detections.append((x, y, box_w, box_h, float(score)))
                    return detections
            except Exception as e:
                logger.error("MediaPipe detection error: %s. Falling back to Haar Cascade.", str(e))

        # Fallback to Haar Cascade detection
        if self.haar_cascade is not None and not self.haar_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, box_w, box_h) in faces:
                detections.append((x, y, box_w, box_h, 0.92))

        return detections
