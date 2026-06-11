from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import DEFAULT_DATASET_DIR, DEFAULT_FEATURE_DIR, DEFAULT_HAND_LANDMARKER_PATH, SplitConfig
from utils.dataset import discover_samples, summarize_dataset, write_manifest
from utils.landmarks import HandFeatureExtractor, extract_media_features
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand-landmark features.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--hand-landmarker-model", type=Path, default=DEFAULT_HAND_LANDMARKER_PATH)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def make_splits(y: np.ndarray, config: SplitConfig) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    relative_val = config.val_size / (1.0 - config.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=config.random_state,
        stratify=y[train_val_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.dataset_dir)
    summary = summarize_dataset(args.dataset_dir, samples)
    labels = summary.labels
    label_to_index = {label: index for index, label in enumerate(labels)}

    LOGGER.info("Extracting features from %d real media samples across %d labels", len(samples), len(labels))
    X: list[np.ndarray] = []
    y: list[int] = []
    kept_samples = []
    skipped = 0

    with HandFeatureExtractor(args.hand_landmarker_model) as extractor:
        for index, sample in enumerate(samples, start=1):
            features = extract_media_features(sample.path, extractor, args.sequence_length, args.frame_stride)
            if features is None:
                skipped += 1
                continue
            X.append(features)
            y.append(label_to_index[sample.label])
            kept_samples.append(sample)
            if index % 100 == 0:
                LOGGER.info("Processed %d/%d samples", index, len(samples))

    if not X:
        raise RuntimeError("Feature extraction produced no usable samples.")

    X_array = np.stack(X).astype(np.float32)
    y_array = np.array(y, dtype=np.int64)
    splits = make_splits(
        y_array,
        SplitConfig(test_size=args.test_size, val_size=args.val_size, random_state=args.random_state),
    )

    np.save(args.output_dir / "X.npy", X_array)
    np.save(args.output_dir / "y.npy", y_array)
    np.savez(args.output_dir / "splits.npz", **splits)
    write_manifest(kept_samples, args.output_dir / "feature_manifest.csv")
    (args.output_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "label_to_index": label_to_index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "feature_config.json").write_text(
        json.dumps(
            {
                "sequence_length": args.sequence_length,
                "feature_dim": int(X_array.shape[-1]),
                "samples_kept": int(X_array.shape[0]),
                "samples_skipped": skipped,
                "splits": {name: int(len(values)) for name, values in splits.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Saved X.npy shape=%s, y.npy shape=%s to %s", X_array.shape, y_array.shape, args.output_dir)
    LOGGER.info("Split sizes: %s", {name: len(values) for name, values in splits.items()})
    LOGGER.info("Skipped %d unreadable/empty samples", skipped)


if __name__ == "__main__":
    main()
