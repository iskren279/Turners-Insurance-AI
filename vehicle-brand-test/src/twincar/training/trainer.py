"""Train/val loop for the flat ``fine_class`` baseline.

Single shared backbone + linear head, cross-entropy loss, AdamW. Logs
train/val loss and val fine/top-3/top-5 plus derived make/model/year accuracy to
W&B (optional), and saves the best checkpoint by ``val/fine_acc`` under
``outputs/checkpoints/``.
"""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from twincar.data.dataset import CarDataset, build_transforms
from twincar.data.label_maps import LabelMaps, load_label_maps
from twincar.preprocess.augmentations import train_augmentation_v1, train_augmentation_v2

from .config import TrainConfig
from .metrics import ClassificationMeter
from .model import build_model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise honour the string."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _train_transform(cfg: TrainConfig):
    """Select the training-image transform from ``cfg.train_aug``.

    Defaults to the original crop+flip ``baseline`` so existing runs are
    unchanged; ``v1`` / ``v2`` opt into the richer car-specific augmentation
    stacks (``v2`` adds occlusion). See ``twincar.preprocess.augmentations``.
    """
    if cfg.train_aug == "v1":
        return train_augmentation_v1(cfg.image_size)
    if cfg.train_aug == "v2":
        return train_augmentation_v2(cfg.image_size)
    if cfg.train_aug not in ("baseline", "", None):
        raise ValueError(f"Unknown train_aug '{cfg.train_aug}'. Choose baseline | v1 | v2.")
    return build_transforms(cfg.image_size, train=True)


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from the split CSVs and dataset root."""
    train_ds = CarDataset(
        cfg.train_csv,
        cfg.root_dir,
        transform=_train_transform(cfg),
        limit=cfg.limit_train,
    )
    val_ds = CarDataset(
        cfg.val_csv,
        cfg.root_dir,
        transform=build_transforms(cfg.image_size, train=False),
        limit=cfg.limit_val,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return the mean per-sample loss."""
    model.train()
    loss_sum = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch = targets.size(0)
        loss_sum += loss.item() * batch
        total += batch
    return loss_sum / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    meter: ClassificationMeter,
) -> dict[str, float]:
    """Run validation and return the computed ``val/...`` metrics dict."""
    model.eval()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        meter.update(logits, targets, loss.item())
    return meter.compute(prefix="val")


def _init_wandb(cfg: TrainConfig):
    """Initialise a W&B run, returning the run handle or ``None`` if disabled."""
    if not cfg.use_wandb:
        return None
    import wandb  # local import so the dep is only needed when logging is on

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=_serialisable_config(cfg),
    )


def _serialisable_config(cfg: TrainConfig) -> dict:
    """Config as plain JSON-friendly values (Paths -> str) for logging/checkpoints."""
    out = {}
    for key, value in asdict(cfg).items():
        out[key] = str(value) if isinstance(value, type(cfg.root_dir)) else value
    return out


def _build_label_mappings(label_maps: LabelMaps) -> list[dict]:
    """Map every classifier output index to its fine_class_name / make / model / year.

    The list is positionally aligned with the model's logits (entry ``i`` ==
    ``fine_class`` ``i``), so a single argmax fully resolves to a productable
    label without re-reading ``labels.yaml`` at inference time.
    """
    mappings: list[dict] = []
    for i in range(label_maps.num_fine_classes):
        year_idx = label_maps.fine_to_year[i]
        mappings.append(
            {
                "fine_class_id": i,
                "fine_class_name": label_maps.fine_labels[i],
                "make": label_maps.makes[label_maps.fine_to_make[i]],
                "model": label_maps.models[label_maps.fine_to_model[i]],
                "year": label_maps.years[year_idx] if year_idx >= 0 else None,
            }
        )
    return mappings


def save_checkpoint(
    path,
    model: nn.Module,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict,
    label_maps: LabelMaps,
) -> None:
    """Save weights plus the full metadata contract for inference/demo loading.

    Writes the canonical keys (``model_state_dict``, ``model_arch``) alongside
    backward-compatible aliases (``model_state``, ``backbone``) so older loaders
    keep working, and adds ``class_names`` + ``label_mappings`` so a checkpoint is
    self-describing — ``predict_batch.py`` and Gradio can load it without guessing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    num_classes = (
        model.get_classifier().out_features
        if hasattr(model, "get_classifier")
        else label_maps.num_fine_classes
    )
    # Ordered by fine_class id, i.e. positionally aligned with the output logits.
    class_names = list(label_maps.fine_labels)
    torch.save(
        {
            "model_state_dict": state_dict,
            "model_state": state_dict,  # backward-compatible alias
            "model_arch": cfg.backbone,
            "backbone": cfg.backbone,  # backward-compatible alias
            "num_classes": num_classes,
            "image_size": cfg.image_size,
            "class_names": class_names,
            "label_mappings": _build_label_mappings(label_maps),
            "epoch": epoch,
            "metrics": metrics,
            "config": _serialisable_config(cfg),
        },
        path,
    )


def fit(cfg: TrainConfig) -> dict:
    """Run the full baseline training loop; return a summary dict.

    Saves the best checkpoint (by ``val/fine_acc``) to
    ``{output_dir}/checkpoints/best.pt`` and returns the best metrics.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    label_maps = load_label_maps(cfg.labels_yaml)
    num_classes = label_maps.num_fine_classes

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"Backbone={cfg.backbone} pretrained={cfg.pretrained} device={device} "
        f"classes={num_classes} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} smoke={cfg.smoke_test}"
    )

    model = build_model(cfg.backbone, num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run = _init_wandb(cfg)
    best_acc = -1.0
    best_metrics: dict = {}
    best_path = cfg.checkpoint_dir / "best.pt"

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            meter = ClassificationMeter(label_maps)
            val_metrics = evaluate(model, val_loader, criterion, device, meter)

            metrics = {"train/loss": train_loss, "epoch": epoch, **val_metrics}
            print(
                f"epoch {epoch}/{cfg.epochs} "
                f"train/loss={train_loss:.4f} "
                f"val/loss={val_metrics['val/loss']:.4f} "
                f"val/fine_acc={val_metrics['val/fine_acc']:.4f}"
            )
            if run is not None:
                run.log(metrics, step=epoch)

            if val_metrics["val/fine_acc"] > best_acc:
                best_acc = val_metrics["val/fine_acc"]
                best_metrics = metrics
                save_checkpoint(best_path, model, cfg, epoch, metrics, label_maps)
                print(f"  saved best checkpoint -> {best_path} (val/fine_acc={best_acc:.4f})")
    finally:
        if run is not None:
            run.finish()

    return {"best_val_fine_acc": best_acc, "best_metrics": best_metrics, "checkpoint": str(best_path)}
