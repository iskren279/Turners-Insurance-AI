"""Read the Stanford Cars MATLAB devkit files.

The devkit ships three ``.mat`` files; this module only reads the two that carry
labels we trust:

* ``cars_meta.mat`` -> ``class_names`` (196 names, indexed by 1-based class id).
* ``cars_train_annos.mat`` -> per-image ``fname`` + ``class`` (1-based class id).

``cars_test_annos.mat`` has no ``class`` field and is intentionally ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import scipy.io as sio


@dataclass(frozen=True)
class TrainAnnotation:
    """A single labelled training image."""

    fname: str
    stanford_class_id: int  # 1-based, as stored in the devkit.


def load_class_names(meta_file: str | Path) -> list[str]:
    """Return class names ordered by 1-based class id (index 0 -> class id 1)."""
    meta_file = Path(meta_file)
    if not meta_file.is_file():
        raise FileNotFoundError(f"Stanford meta file not found: {meta_file}")
    mat = sio.loadmat(str(meta_file))
    class_names = mat["class_names"][0]
    return [str(name[0]).strip() for name in class_names]


def load_train_annotations(annotation_file: str | Path) -> list[TrainAnnotation]:
    """Return labelled training annotations from ``cars_train_annos.mat``."""
    annotation_file = Path(annotation_file)
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Stanford train annotations not found: {annotation_file}")
    mat = sio.loadmat(str(annotation_file))
    records = mat["annotations"][0]
    if "class" not in records.dtype.names:
        raise ValueError(
            f"{annotation_file} has no 'class' field; it cannot be used as labelled data."
        )
    annotations: list[TrainAnnotation] = []
    for rec in records:
        annotations.append(
            TrainAnnotation(
                fname=str(rec["fname"][0]).strip(),
                stanford_class_id=int(rec["class"][0][0]),
            )
        )
    return annotations
