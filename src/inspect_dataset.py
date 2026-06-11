from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.config import DEFAULT_DATASET_DIR, DEFAULT_REPORT_DIR
from utils.dataset import discover_samples, summarize_dataset, write_dataset_report, write_manifest
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an AzSLD-style dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    samples = discover_samples(args.dataset_dir)
    summary = summarize_dataset(args.dataset_dir, samples)
    json_path, md_path = write_dataset_report(summary, args.report_dir)
    write_manifest(samples, args.report_dir / "dataset_manifest.csv")

    LOGGER.info("Dataset organization: %s", summary.organization)
    LOGGER.info("Detected %d samples across %d labels", summary.total_samples, summary.total_classes)
    LOGGER.info("File formats: %s", summary.file_formats)
    LOGGER.info("Report written to %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()

