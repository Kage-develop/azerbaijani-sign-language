from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import DEFAULT_HAND_LANDMARKER_PATH, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, PROJECT_ROOT
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_TFLITE_PATH = (
    PROJECT_ROOT / "mobile" / "android" / "app" / "src" / "main" / "assets_runtime" / "sign_model.tflite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained Keras model to mobile TFLite format.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_TFLITE_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--hand-landmarker-path", type=Path, default=DEFAULT_HAND_LANDMARKER_PATH)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install dependencies with: pip install -r requirements.txt") from exc

    if not args.model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {args.model_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    if not args.hand_landmarker_path.exists():
        raise FileNotFoundError(f"Hand landmarker file not found: {args.hand_landmarker_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model_path)
    input_shape = model.inputs[0].shape
    sequence_length = int(input_shape[1])
    feature_dim = int(input_shape[2])

    @tf.function(input_signature=[tf.TensorSpec([1, sequence_length, feature_dim], tf.float32)])
    def serving_fn(inputs: tf.Tensor) -> tf.Tensor:
        return model(inputs, training=False)

    concrete_function = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function], model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    args.output_path.write_bytes(tflite_model)
    shutil.copy2(args.labels_path, args.output_path.parent / "labels.json")
    shutil.copy2(args.hand_landmarker_path, args.output_path.parent / "hand_landmarker.task")
    LOGGER.info("Saved TFLite model to %s (%d bytes)", args.output_path, len(tflite_model))
    LOGGER.info("Copied mobile labels and hand landmarker assets to %s", args.output_path.parent)


if __name__ == "__main__":
    main()
