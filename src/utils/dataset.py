from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    extension: str
    kind: str


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: str
    organization: str
    total_classes: int
    total_samples: int
    file_formats: dict[str, int]
    class_counts: dict[str, int]
    min_samples_per_class: int
    max_samples_per_class: int
    mean_samples_per_class: float
    labels: list[str]


def iter_supported_files(path: Path) -> Iterable[Path]:
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def discover_samples(dataset_dir: Path) -> list[Sample]:
    """Discover labels dynamically from first-level folders containing media files."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    samples: list[Sample] = []
    for class_dir in sorted((p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        label = class_dir.name
        for media_path in sorted(iter_supported_files(class_dir), key=lambda p: str(p)):
            ext = media_path.suffix.lower()
            kind = "video" if ext in VIDEO_EXTENSIONS else "image"
            samples.append(Sample(path=media_path, label=label, extension=ext, kind=kind))

    if not samples:
        raise ValueError(f"No supported media files found under {dataset_dir}")

    return samples


def summarize_dataset(dataset_dir: Path, samples: list[Sample] | None = None) -> DatasetSummary:
    samples = samples or discover_samples(dataset_dir)
    class_counts = Counter(sample.label for sample in samples)
    file_formats = Counter(sample.extension for sample in samples)
    labels = sorted(class_counts)
    counts = list(class_counts.values())
    return DatasetSummary(
        dataset_dir=str(dataset_dir.resolve()),
        organization="class-per-folder; label is inferred from the first-level directory name",
        total_classes=len(labels),
        total_samples=len(samples),
        file_formats=dict(sorted(file_formats.items())),
        class_counts=dict(sorted(class_counts.items())),
        min_samples_per_class=min(counts),
        max_samples_per_class=max(counts),
        mean_samples_per_class=round(sum(counts) / len(counts), 2),
        labels=labels,
    )


def write_dataset_report(summary: DatasetSummary, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "dataset_report.json"
    md_path = report_dir / "dataset_report.md"

    json_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# AzSLD Dataset Report",
        "",
        f"- Dataset path: `{summary.dataset_dir}`",
        f"- Organization: {summary.organization}",
        f"- Total classes: {summary.total_classes}",
        f"- Total samples: {summary.total_samples}",
        f"- File formats: {summary.file_formats}",
        f"- Samples per class: min={summary.min_samples_per_class}, "
        f"max={summary.max_samples_per_class}, mean={summary.mean_samples_per_class}",
        "",
        "## Class Counts",
        "",
        "| Label | Samples |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {count} |" for label, count in summary.class_counts.items())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def write_manifest(samples: list[Sample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "extension", "kind"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": str(sample.path),
                    "label": sample.label,
                    "extension": sample.extension,
                    "kind": sample.kind,
                }
            )

