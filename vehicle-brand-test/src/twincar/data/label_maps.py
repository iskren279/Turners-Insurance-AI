"""Turn ``configs/labels.yaml`` into integer index maps for training.

``labels.yaml`` (written by ``scripts/prepare_data.py``) is the single source of
truth for class counts and the ``fine_class -> make/model/year`` relationship.
This module loads it into the index maps the training loop needs, so make / model
/ year accuracy can be *derived* from a flat ``fine_class`` prediction without
adding separate heads (those come later).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LabelMaps:
    """Index maps derived from ``labels.yaml``.

    The ``fine_to_*`` lists are indexed by ``fine_class`` (0-based) and hold the
    index of that class's make / model / year within :attr:`makes`,
    :attr:`models`, and :attr:`years` respectively.
    """

    fine_labels: list[str]
    makes: list[str]
    models: list[str]
    years: list[int]
    fine_to_make: list[int]
    fine_to_model: list[int]
    fine_to_year: list[int]

    @property
    def num_fine_classes(self) -> int:
        return len(self.fine_labels)

    @property
    def num_makes(self) -> int:
        return len(self.makes)

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def num_years(self) -> int:
        return len(self.years)


def load_label_maps(labels_yaml_path: str | Path) -> LabelMaps:
    """Load :class:`LabelMaps` from a ``labels.yaml`` file.

    The ``fine_classes`` list is sorted by ``fine_class`` so the resulting maps
    are positionally aligned with the model's output logits.
    """
    path = Path(labels_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"labels.yaml not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    fine_entries = sorted(payload.get("fine_classes", []), key=lambda e: e["fine_class"])
    if not fine_entries:
        raise ValueError(f"{path} has no fine_classes entries")

    makes: list[str] = list(payload.get("makes", []))
    models: list[str] = list(payload.get("models", []))
    years: list[int] = list(payload.get("years", []))
    make_index = {name: i for i, name in enumerate(makes)}
    model_index = {name: i for i, name in enumerate(models)}
    year_index = {value: i for i, value in enumerate(years)}

    fine_labels: list[str] = []
    fine_to_make: list[int] = []
    fine_to_model: list[int] = []
    fine_to_year: list[int] = []
    for expected, entry in enumerate(fine_entries):
        if entry["fine_class"] != expected:
            raise ValueError(
                f"{path} fine_class values are not contiguous from 0 "
                f"(expected {expected}, got {entry['fine_class']})"
            )
        fine_labels.append(entry["fine_label"])
        fine_to_make.append(make_index[entry["make"]])
        fine_to_model.append(model_index[entry["model"]])
        # Year may be missing for some classes; map to -1 so equality never spuriously matches.
        year = entry.get("year")
        fine_to_year.append(year_index.get(year, -1) if year is not None else -1)

    return LabelMaps(
        fine_labels=fine_labels,
        makes=makes,
        models=models,
        years=years,
        fine_to_make=fine_to_make,
        fine_to_model=fine_to_model,
        fine_to_year=fine_to_year,
    )
