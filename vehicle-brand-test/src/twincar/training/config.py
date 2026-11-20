"""Resolve the baseline training configuration.

Settings come from ``configs/base.yaml`` (a plain YAML file). ``--smoke-test``
layers on fast, offline, CPU-safe defaults, and explicit CLI overrides always win
over both. The dataset root is resolved separately from
``configs/datasets.local.yaml`` and passed in as ``root_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Smoke-test defaults: tiny, untrained, single-process, no network. A small
# offline-constructable backbone keeps the run fast on CPU and CI.
SMOKE_BACKBONE = "resnet18"
SMOKE_IMAGE_SIZE = 64
SMOKE_BATCH_SIZE = 2
SMOKE_EPOCHS = 1
SMOKE_LIMIT = 8


@dataclass
class TrainConfig:
    """Everything :func:`twincar.training.trainer.fit` needs for one run."""

    # Data
    root_dir: Path = Path("")
    train_csv: Path = Path("splits/stanford_train.csv")
    val_csv: Path = Path("splits/stanford_val.csv")
    labels_yaml: Path = Path("configs/labels.yaml")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_aug: str = "baseline"  # baseline (crop+flip) | v1 | v2 (v1 + occlusion)

    # Model
    backbone: str = "convnextv2_base"
    pretrained: bool = True

    # Optimisation
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 0.05

    # Logging
    use_wandb: bool = True
    wandb_project: str = "twincar"
    wandb_entity: str | None = None

    # Runtime
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("outputs")
    limit_train: int | None = None
    limit_val: int | None = None
    smoke_test: bool = False

    # Loss weighting placeholder for the future hybrid heads.
    # TODO(hybrid): add make/model/year heads and a weighted multi-task loss.
    metadata: dict = field(default_factory=dict)

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_train_config(
    config_path: str | Path,
    root_dir: str | Path,
    *,
    smoke_test: bool = False,
    overrides: dict | None = None,
) -> TrainConfig:
    """Build a :class:`TrainConfig` from ``base.yaml`` plus smoke/CLI layers.

    Precedence (lowest to highest): base.yaml values, then smoke-test defaults
    (only when ``smoke_test`` is set), then explicit ``overrides`` whose value is
    not ``None``.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    project = raw.get("project", {})
    data = raw.get("data", {})
    training = raw.get("training", {})
    model = raw.get("model", {})
    logging = raw.get("logging", {})

    cfg = TrainConfig(
        root_dir=_as_path(root_dir),
        image_size=int(data.get("image_size", 224)),
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(data.get("num_workers", 4)),
        train_aug=str(data.get("train_aug", "baseline")),
        backbone=str(model.get("backbone", "convnextv2_base")),
        pretrained=bool(model.get("pretrained", True)),
        epochs=int(training.get("epochs", 25)),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.05)),
        use_wandb=bool(logging.get("use_wandb", True)),
        wandb_project=str(logging.get("wandb_project", "twincar")),
        wandb_entity=logging.get("wandb_entity"),
        seed=int(project.get("seed", 42)),
        smoke_test=smoke_test,
    )

    if smoke_test:
        cfg.backbone = SMOKE_BACKBONE
        cfg.pretrained = False
        cfg.image_size = SMOKE_IMAGE_SIZE
        cfg.batch_size = SMOKE_BATCH_SIZE
        cfg.epochs = SMOKE_EPOCHS
        cfg.num_workers = 0
        cfg.device = "cpu"
        cfg.use_wandb = False
        cfg.limit_train = SMOKE_LIMIT
        cfg.limit_val = SMOKE_LIMIT

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown TrainConfig override: {key}")
        current = getattr(cfg, key)
        setattr(cfg, key, _as_path(value) if isinstance(current, Path) else value)

    return cfg
