from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np

from utils.config import DEFAULT_HAND_LANDMARKER_PATH, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, FEATURE_DIM
from utils.landmarks import HandFeatureExtractor
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

AZ_TO_ASCII = str.maketrans(
    {
        "Ə": "E",
        "ə": "e",
        "İ": "I",
        "ı": "i",
        "Ş": "S",
        "ş": "s",
        "Ü": "U",
        "ü": "u",
        "Ç": "C",
        "ç": "c",
        "Ğ": "G",
        "ğ": "g",
        "Ö": "O",
        "ö": "o",
    }
)


def to_display_label(label: str) -> str:
    """Return an OpenCV-friendly ASCII label for webcam overlay text."""
    return label.translate(AZ_TO_ASCII)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam AzSL word inference.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--hand-landmarker-model", type=Path, default=DEFAULT_HAND_LANDMARKER_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        import cv2
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")

    labels_payload = json.loads(args.labels_path.read_text(encoding="utf-8"))
    labels = labels_payload["labels"]
    model = tf.keras.models.load_model(args.model_path)
    sequence = deque(maxlen=args.sequence_length)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    LOGGER.info("Starting webcam inference. Press 'q' to quit.")
    with HandFeatureExtractor(args.hand_landmarker_model) as extractor:
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    LOGGER.warning("Webcam frame read failed")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                feature = extractor.frame_to_feature(frame_rgb)
                if feature.shape[0] != FEATURE_DIM:
                    LOGGER.warning("Unexpected feature size: %s", feature.shape)
                    continue
                sequence.append(feature)

                display_label = "Collecting..."
                confidence = 0.0
                if len(sequence) == args.sequence_length:
                    X = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                    probabilities = model.predict(X, verbose=0)[0]
                    class_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[class_index])
                    display_label = (
                        to_display_label(labels[class_index])
                        if confidence >= args.confidence_threshold
                        else "Unknown"
                    )

                cv2.rectangle(frame_bgr, (10, 10), (560, 80), (0, 0, 0), thickness=-1)
                cv2.putText(
                    frame_bgr,
                    f"{display_label} ({confidence:.2%})",
                    (24, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if display_label != "Unknown" else (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SignBridge AzSL Word Recognition", frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
