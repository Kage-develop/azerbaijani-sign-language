from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "AzSLD_Words_100"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sign_model.h5"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
DEFAULT_HAND_LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = MAX_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
