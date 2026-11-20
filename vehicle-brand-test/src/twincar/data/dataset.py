"""Dataset and transforms for the flat ``fine_class`` baseline.

:class:`CarDataset` reads one of the generated split CSVs and loads images by
joining the dataset-root-relative ``image_path`` column to the dataset root from
``configs/datasets.local.yaml``. It yields ``(image_tensor, fine_class)`` pairs —
the make / model / year columns are intentionally not returned here because the
baseline trains a single fine head and derives the rest via
:mod:`twincar.data.label_maps`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet statistics; timm pretrained backbones expect this normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    """Build train (augmented) or eval (deterministic) image transforms."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.14))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _read_split_rows(csv_path: Path) -> list[tuple[str, int]]:
    """Return ``(image_path, fine_class)`` rows from a split CSV."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"image_path", "fine_class"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        return [(row["image_path"], int(row["fine_class"])) for row in reader]


class CarDataset(Dataset):
    """Image-classification dataset backed by a TwinCar split CSV.

    Args:
        csv_path: Path to a split CSV (``stanford_train.csv`` / ``stanford_val.csv``).
        root_dir: Dataset root that the CSV's relative ``image_path`` values join to.
        transform: Optional torchvision transform applied to each PIL image.
        limit: If set, keep only the first ``limit`` rows (used by the smoke test).
    """

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
        limit: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = _read_split_rows(self.csv_path)
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel_path, fine_class = self.samples[index]
        image = Image.open(self.root_dir / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, fine_class

    @property
    def num_classes_seen(self) -> int:
        """Number of distinct ``fine_class`` values present in this split."""
        return len({label for _, label in self.samples})
