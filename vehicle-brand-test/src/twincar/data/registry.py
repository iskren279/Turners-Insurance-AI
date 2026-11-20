"""Build the TwinCar label registry and train/val splits from Stanford Cars.

Produces, from parsed devkit data:

* ``stanford_labels.csv`` — every labelled training image with its split.
* ``stanford_train.csv`` / ``stanford_val.csv`` — the split subsets.
* ``labels.yaml`` — the make/model/year/fine_class mapping (counts derived here,
  never hardcoded downstream).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from .labels import parse_class_name
from .stanford import TrainAnnotation

SOURCE_STANFORD = "stanford_cars"

# Column order for every generated CSV.
CSV_COLUMNS = (
    "image_path",
    "source",
    "fine_class",
    "stanford_class_id",
    "fine_label",
    "make",
    "model",
    "year",
    "split",
)


@dataclass
class LabelRecord:
    """One labelled image row, matching :data:`CSV_COLUMNS`."""

    image_path: str
    source: str
    fine_class: int
    stanford_class_id: int
    fine_label: str
    make: str
    model: str
    year: int | None
    split: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def build_label_records(
    class_names: list[str],
    annotations: list[TrainAnnotation],
    train_images_rel: str,
) -> list[LabelRecord]:
    """Join annotations with parsed class names into label records.

    ``image_path`` is a dataset-root-relative POSIX path (``train_images_rel`` +
    filename) so the CSVs never leak absolute machine paths.
    """
    parsed = [parse_class_name(name) for name in class_names]
    records: list[LabelRecord] = []
    for ann in annotations:
        class_id = ann.stanford_class_id
        if not 1 <= class_id <= len(class_names):
            raise ValueError(
                f"Annotation {ann.fname} has class id {class_id} outside 1..{len(class_names)}"
            )
        info = parsed[class_id - 1]
        records.append(
            LabelRecord(
                image_path=f"{train_images_rel}/{ann.fname}",
                source=SOURCE_STANFORD,
                fine_class=class_id - 1,  # 0-based ML label
                stanford_class_id=class_id,  # original 1-based id
                fine_label=info.fine_label,
                make=info.make,
                model=info.model,
                year=info.year,
                split="",
            )
        )
    return records


def _can_stratify(labels: list[int], val_size: float) -> bool:
    """Stratification needs >=2 per class and >=1 val slot per class."""
    if not labels:
        return False
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if min(counts.values()) < 2:
        return False
    n_classes = len(counts)
    n_val = int(round(len(labels) * val_size))
    return n_val >= n_classes and (len(labels) - n_val) >= n_classes


def assign_splits(
    records: list[LabelRecord],
    val_size: float = 0.2,
    seed: int = 42,
) -> str:
    """Assign ``split`` to each record in place; return the strategy used.

    Prefers a deterministic stratified split by ``fine_class`` (seed 42), and
    falls back to a plain deterministic shuffle split when class counts are too
    small to stratify (e.g. tiny test fixtures).
    """
    if not records:
        return "empty"
    indices = list(range(len(records)))
    labels = [rec.fine_class for rec in records]
    stratify = labels if _can_stratify(labels, val_size) else None
    _, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    val_set = set(val_idx)
    for i, rec in enumerate(records):
        rec.split = "val" if i in val_set else "train"
    return "stratified" if stratify is not None else "random"


def write_csv(records: list[LabelRecord], path: str | Path) -> None:
    """Write records to ``path`` in :data:`CSV_COLUMNS` order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.as_row())


@dataclass
class WriteResult:
    """Summary of a registry generation run."""

    labels_csv: Path
    train_csv: Path
    val_csv: Path
    labels_yaml: Path
    n_total: int
    n_train: int
    n_val: int
    n_classes: int
    split_strategy: str = field(default="")


def build_labels_yaml(class_names: list[str]) -> dict:
    """Build the labels.yaml payload with counts derived from parsed data."""
    fine_classes: list[dict] = []
    makes: set[str] = set()
    models: set[str] = set()
    years: set[int] = set()
    for idx, name in enumerate(class_names):
        info = parse_class_name(name)
        fine_classes.append(
            {
                "fine_class": idx,
                "stanford_class_id": idx + 1,
                "fine_label": info.fine_label,
                "make": info.make,
                "model": info.model,
                "year": info.year,
            }
        )
        makes.add(info.make)
        models.add(info.model)
        if info.year is not None:
            years.add(info.year)
    return {
        "num_fine_classes": len(class_names),
        "fine_classes": fine_classes,
        "makes": sorted(makes),
        "models": sorted(models),
        "years": sorted(years),
    }


def write_labels_yaml(class_names: list[str], path: str | Path) -> None:
    """Write the labels.yaml mapping to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_labels_yaml(class_names)
    header = (
        "# Generated by scripts/prepare_data.py.\n"
        "# Source of truth for make/model/year/fine_class mappings.\n"
        "# Do not hardcode class counts in model code.\n"
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def generate_registry(
    class_names: list[str],
    annotations: list[TrainAnnotation],
    train_images_rel: str,
    splits_dir: str | Path,
    labels_yaml_path: str | Path,
    val_size: float = 0.2,
    seed: int = 42,
) -> WriteResult:
    """End-to-end: build records, split, and write all four output files."""
    records = build_label_records(class_names, annotations, train_images_rel)
    strategy = assign_splits(records, val_size=val_size, seed=seed)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]

    splits_dir = Path(splits_dir)
    labels_csv = splits_dir / "stanford_labels.csv"
    train_csv = splits_dir / "stanford_train.csv"
    val_csv = splits_dir / "stanford_val.csv"

    write_csv(records, labels_csv)
    write_csv(train_records, train_csv)
    write_csv(val_records, val_csv)
    write_labels_yaml(class_names, labels_yaml_path)

    return WriteResult(
        labels_csv=labels_csv,
        train_csv=train_csv,
        val_csv=val_csv,
        labels_yaml=Path(labels_yaml_path),
        n_total=len(records),
        n_train=len(train_records),
        n_val=len(val_records),
        n_classes=len(class_names),
        split_strategy=strategy,
    )
