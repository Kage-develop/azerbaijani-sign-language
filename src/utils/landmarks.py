from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import FEATURE_DIM, LANDMARKS_PER_HAND, MAX_HANDS, VIDEO_EXTENSIONS

LOGGER = logging.getLogger(__name__)


class HandFeatureExtractor:
    """MediaPipe Tasks based two-hand landmark feature extractor."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                "MediaPipe hand landmarker model not found: "
                f"{model_path}. Download hand_landmarker.task into the models folder."
            )

        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandFeatureExtractor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def frame_to_feature(self, frame_rgb: np.ndarray) -> np.ndarray:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect(image)
        if not getattr(results, "hand_landmarks", None):
            return _empty_frame_features()

        hand_blocks: list[tuple[float, np.ndarray]] = []
        handedness = getattr(results, "handedness", None) or []
        for index, hand_landmarks in enumerate(results.hand_landmarks[:MAX_HANDS]):
            score = 0.0
            if index < len(handedness) and handedness[index]:
                score = float(getattr(handedness[index][0], "score", 0.0))

            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )
            coords = normalize_hand(coords)
            hand_blocks.append((score, coords.reshape(-1)))

        return pack_hand_blocks(hand_blocks)


def _empty_frame_features() -> np.ndarray:
    return np.zeros((FEATURE_DIM,), dtype=np.float32)


def pack_hand_blocks(hand_blocks: list[tuple[float, np.ndarray]]) -> np.ndarray:
    feature = _empty_frame_features()
    hand_size = LANDMARKS_PER_HAND * 3
    hand_blocks.sort(key=lambda item: item[0], reverse=True)
    for hand_index, (_, block) in enumerate(hand_blocks[:MAX_HANDS]):
        start = hand_index * hand_size
        feature[start : start + hand_size] = block
    return feature


def normalize_hand(coords: np.ndarray) -> np.ndarray:
    """Normalize landmarks relative to wrist and hand scale for better generalization."""
    normalized = coords.copy()
    normalized -= normalized[0]
    scale = np.linalg.norm(normalized, axis=1).max()
    if scale > 1e-6:
        normalized /= scale
    return normalized


def sample_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    if sequence.shape[0] == 0:
        return np.zeros((sequence_length, FEATURE_DIM), dtype=np.float32)
    if sequence.shape[0] == sequence_length:
        return sequence.astype(np.float32)
    indices = np.linspace(0, sequence.shape[0] - 1, sequence_length).astype(int)
    return sequence[indices].astype(np.float32)


def extract_media_features(
    media_path: Path,
    extractor: HandFeatureExtractor,
    sequence_length: int,
    frame_stride: int = 1,
) -> np.ndarray | None:
    """Extract fixed-length MediaPipe hand landmark sequence from a video or image."""
    import cv2

    suffix = media_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            LOGGER.warning("Could not open video: %s", media_path)
            return None

        features: list[np.ndarray] = []
        frame_index = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    features.append(extractor.frame_to_feature(frame_rgb))
                frame_index += 1
        finally:
            capture.release()

        if not features:
            LOGGER.warning("No frames decoded from video: %s", media_path)
            return None
        return sample_sequence(np.vstack(features), sequence_length)

    frame_bgr = cv2.imread(str(media_path))
    if frame_bgr is None:
        LOGGER.warning("Could not open image: %s", media_path)
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return sample_sequence(np.expand_dims(extractor.frame_to_feature(frame_rgb), axis=0), sequence_length)
