"""Load dataset path configuration from ``configs/datasets.local.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StanfordPaths:
    """Resolved filesystem locations for the Stanford Cars dataset.

    ``train_images_rel`` is kept as a dataset-root-relative POSIX path because it
    becomes the prefix for the relative ``image_path`` written to the CSVs.
    """

    root_dir: Path
    meta_file: Path
    train_annotation_file: Path
    train_images_rel: str

    @property
    def train_images_dir(self) -> Path:
        return self.root_dir / self.train_images_rel


def load_dataset_config(config_path: str | Path, dataset_key: str = "stanford_cars") -> dict:
    """Read the YAML config and return the section for ``dataset_key``."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if dataset_key not in config:
        raise KeyError(f"Dataset key '{dataset_key}' missing from {config_path}")
    return config[dataset_key]


def resolve_stanford_paths(config_path: str | Path) -> StanfordPaths:
    """Build :class:`StanfordPaths` from the ``stanford_cars`` config section."""
    section = load_dataset_config(config_path, "stanford_cars")
    root_dir = Path(section["root_dir"])
    meta_file = root_dir / section["meta_file"]
    train_annotation_file = root_dir / section["train_annotation_file"]
    # Normalise to a POSIX-style relative path for use as the CSV path prefix.
    train_images_rel = Path(section["train_images_dir"]).as_posix()
    return StanfordPaths(
        root_dir=root_dir,
        meta_file=meta_file,
        train_annotation_file=train_annotation_file,
        train_images_rel=train_images_rel,
    )
