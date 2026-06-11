from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from utils.config import DEFAULT_FEATURE_DIR, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM AzSL word classifier.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_model(input_shape: tuple[int, int], num_classes: int, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install dependencies with: pip install -r requirements.txt") from exc

    X = np.load(args.features_dir / "X.npy")
    y = np.load(args.features_dir / "y.npy")
    splits = np.load(args.features_dir / "splits.npz")
    labels_payload = json.loads((args.features_dir / "labels.json").read_text(encoding="utf-8"))
    labels = labels_payload["labels"]

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.labels_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model((X.shape[1], X.shape[2]), len(labels), args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    LOGGER.info("Training on %d samples, validating on %d, testing on %d", len(train_idx), len(val_idx), len(test_idx))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if not args.model_path.exists():
        model.save(args.model_path)

    best_model = tf.keras.models.load_model(args.model_path)
    probabilities = best_model.predict(X_test, batch_size=args.batch_size, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": labels,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }

    metrics_path = args.model_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    args.labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved model to %s", args.model_path)
    LOGGER.info("Saved labels to %s", args.labels_path)
    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    LOGGER.info(
        "Test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )


if __name__ == "__main__":
    main()

