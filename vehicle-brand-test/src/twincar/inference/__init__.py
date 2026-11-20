"""
src/twincar/inference/__init__.py

Central inference module — the single source of truth for model loading
and prediction. Both predict_batch.py and gradio_app.py import from here.

Usage:
    from twincar.inference import load_bundle, predict_single, predict_batch, PredictionResult

Stub mode (no checkpoint):
    bundle = load_bundle(checkpoint=None)
    result = predict_single(image, bundle)   # returns prediction="PENDING"

Real mode:
    bundle = load_bundle(checkpoint=Path("model.pt"), device="cuda")
    result = predict_single(image, bundle)   # returns real prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class TopKEntry:
    label:      str
    confidence: float


@dataclass
class PredictionResult:
    """
    Structured output for a single image prediction.
    Fields are always present; empty strings / 0.0 used when not available.
    """
    filename:   str
    filepath:   str
    prediction: str           # full label e.g. "BMW 3 Series 2012", "PENDING", "ERROR"
    confidence: float         # 0.0–1.0
    make:       str           # parsed from prediction
    model:      str           # parsed from prediction
    year:       str           # parsed from prediction, "" if absent
    top_k:      list[TopKEntry] = field(default_factory=list)
    error:      str           = ""

    def to_dict(self) -> dict:
        """Flat dict for CSV writing — mirrors predict_batch.py OUTPUT_FIELDNAMES."""
        return {
            "filename":          self.filename,
            "filepath":          self.filepath,
            "prediction":        self.prediction,
            "confidence":        round(self.confidence, 4),
            "make":              self.make,
            "model":             self.model,
            "year":              self.year,
            "top_k_predictions": " | ".join(
                f"{e.label} ({e.confidence:.3f})" for e in self.top_k
            ),
            "error":             self.error,
        }


@dataclass
class ModelBundle:
    """
    Holds the loaded model + metadata needed at inference time.
    In stub mode, model is None and class_names is empty.

    ``label_mappings`` (when present) is indexed by classifier output index and
    holds the authoritative make/model/year for each fine class, so predictions
    don't have to parse them out of the label string.
    """
    model:       object | None      # torch.nn.Module or None
    class_names: list[str]
    device:      str
    image_size:  int
    stub:        bool               # True when no checkpoint was loaded
    label_mappings: list[dict] = field(default_factory=list)


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_label(label: str) -> tuple[str, str, str]:
    """
    Split a class label string into (make, model, year).

    Handles:
        "BMW 3 Series 2012"  → ("BMW", "3 Series", "2012")
        "BMW_3_Series_2012"  → same
        "BMW 3 Series"       → ("BMW", "3 Series", "")
        "Toyota"             → ("Toyota", "", "")
    """
    label = label.replace("_", " ").strip()
    parts = label.split()

    year = ""
    if parts and parts[-1].isdigit() and 1900 <= int(parts[-1]) <= 2030:
        year  = parts[-1]
        parts = parts[:-1]

    make      = parts[0] if parts else ""
    car_model = " ".join(parts[1:]) if len(parts) > 1 else ""
    return make, car_model, year


def _resolve_fields(idx: int, label: str, bundle: ModelBundle) -> tuple[str, str, str]:
    """Return (make, model, year) for a predicted class index.

    Prefers the checkpoint's authoritative ``label_mappings`` (keyed by output
    index); falls back to parsing the label string for older checkpoints that
    don't carry mappings.
    """
    mappings = bundle.label_mappings
    if mappings and 0 <= idx < len(mappings):
        entry = mappings[idx]
        year = entry.get("year")
        return entry.get("make", ""), entry.get("model", ""), "" if year is None else str(year)
    return parse_label(label)


# ── Transform ─────────────────────────────────────────────────────────────────

def _build_transform(image_size: int):
    """Build the torchvision inference transform.

    Mirrors the validation/evaluation preprocessing
    (``twincar.data.dataset.build_transforms(..., train=False)``):
    aspect-preserving ``Resize`` to ``round(image_size * 1.14)`` followed by a
    ``CenterCrop`` to ``image_size``, so deployed inference (API / Gradio /
    batch) sees the same geometry the model was validated on. A direct
    ``Resize((image_size, image_size))`` would distort aspect ratio.

    Imported lazily to keep stub mode torch-free.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(round(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_bundle(
    checkpoint: Path | None = None,
    device: str = "auto",
    image_size: int = IMAGE_SIZE,
) -> ModelBundle:
    """
    Load model and metadata from checkpoint file.

    If checkpoint is None, returns a stub ModelBundle (no torch required).
    Device "auto" selects cuda > mps > cpu automatically.

    Accepts both the current metadata contract and older checkpoints:
        weights:    "model_state_dict"  OR "model_state"  OR a raw state dict
        arch:       "model_arch"        OR "backbone"     OR config["backbone"]
        image_size: "image_size"        OR config["image_size"]  OR default
        labels:     "class_names" (preferred) OR derived from "label_mappings"

    See ``twincar.training.trainer.save_checkpoint`` for the canonical writer.
    """
    if checkpoint is None:
        log.info("No checkpoint provided — running in stub mode.")
        return ModelBundle(
            model=None,
            class_names=[],
            device="cpu",
            image_size=image_size,
            stub=True,
            label_mappings=[],
        )

    # ── Resolve device ────────────────────────────────────────────────────────
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    resolved = torch.device(device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log.info("Loading checkpoint '%s' onto %s", checkpoint, resolved)
    ckpt = torch.load(checkpoint, map_location=resolved)

    config = ckpt.get("config", {}) or {}

    # Labels: prefer explicit class_names, else derive from label_mappings (both
    # are ordered by fine_class id == output index).
    label_mappings: list[dict] = ckpt.get("label_mappings", []) or []
    class_names: list[str] = ckpt.get("class_names", [])
    if not class_names and label_mappings:
        class_names = [m.get("fine_class_name", "") for m in label_mappings]
    if not class_names:
        raise KeyError(
            f"Checkpoint '{checkpoint}' is missing 'class_names'. "
            "Re-save it with the current metadata contract "
            "(twincar.training.trainer.save_checkpoint)."
        )

    # Accept both new and old key names for arch / image_size / weights.
    arch = (
        ckpt.get("model_arch")
        or ckpt.get("backbone")
        or config.get("backbone")
        or "convnext_tiny"
    )
    img_size = ckpt.get("image_size") or config.get("image_size") or image_size
    num_classes = len(class_names)

    model = _build_model(arch, num_classes)
    state = ckpt.get("model_state_dict")
    if state is None:
        state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(resolved)
    model.eval()
    log.info("Model ready — %d classes, arch=%s, device=%s", num_classes, arch, resolved)

    return ModelBundle(
        model=model,
        class_names=class_names,
        device=str(resolved),
        image_size=img_size,
        stub=False,
        label_mappings=label_mappings,
    )


def _build_model(arch: str, num_classes: int):
    """Try timm first, fall back to torchvision."""
    import torch.nn as nn

    try:
        import timm
        return timm.create_model(arch, pretrained=False, num_classes=num_classes)
    except Exception:
        pass

    from torchvision import models
    arch_map = {
        "convnext_tiny":   models.convnext_tiny,
        "convnext_small":  models.convnext_small,
        "convnext_base":   models.convnext_base,
        "resnet18":        models.resnet18,   # used in pipeline sanity runs
        "resnet34":        models.resnet34,
        "resnet50":        models.resnet50,
        "efficientnet_b3": models.efficientnet_b3,
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(arch_map)}")

    m = arch_map[arch](weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif hasattr(m, "classifier"):
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif hasattr(m, "head"):
        m.head = nn.Linear(m.head.in_features, num_classes)
    return m


# ── Single-image prediction ───────────────────────────────────────────────────

def predict_single(
    image: Image.Image,
    bundle: ModelBundle,
    top_k: int = 3,
    source_path: str = "",
) -> PredictionResult:
    """
    Run inference on a single PIL image.

    Works in stub mode (returns PENDING) and real mode.
    This is the function Gradio calls — one image at a time.

    NOTE: Currently assumes flat fine-class output (single logits tensor).
    If Pair A ships the 4-head architecture (separate make/model/year heads),
    this function needs to be updated to unpack multiple outputs.
    Coordinate with Pair A before Day 5.
    """
    filename = Path(source_path).name if source_path else "unknown"

    if bundle.stub:
        return PredictionResult(
            filename=filename,
            filepath=source_path,
            prediction="PENDING",
            confidence=0.0,
            make="", model="", year="",
            top_k=[],
        )

    import torch
    import torch.nn.functional as F

    transform = _build_transform(bundle.image_size)
    device    = torch.device(bundle.device)

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = bundle.model(tensor)
        probs  = F.softmax(logits, dim=1)
        k      = min(top_k, len(bundle.class_names))
        topk_probs, topk_idx = probs.topk(k, dim=1)

    top1_idx   = topk_idx[0, 0].item()
    top1_label = bundle.class_names[top1_idx]
    top1_conf  = topk_probs[0, 0].item()
    make, car_model, year = _resolve_fields(top1_idx, top1_label, bundle)

    top_k_entries = [
        TopKEntry(
            label=bundle.class_names[topk_idx[0, i].item()],
            confidence=topk_probs[0, i].item(),
        )
        for i in range(k)
    ]

    return PredictionResult(
        filename=filename,
        filepath=source_path,
        prediction=top1_label,
        confidence=top1_conf,
        make=make,
        model=car_model,
        year=year,
        top_k=top_k_entries,
    )


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(
    image_paths: list[Path],
    bundle: ModelBundle,
    batch_size: int = 32,
    top_k: int = 1,
) -> list[PredictionResult]:
    """
    Run inference on a list of image paths in batches.

    This is what predict_batch.py calls internally.
    Stub mode returns PENDING for all images without loading torch.
    """
    if bundle.stub:
        results = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                results.append(predict_single(img, bundle, source_path=str(path)))
            except Exception as exc:
                results.append(PredictionResult(
                    filename=path.name, filepath=str(path),
                    prediction="ERROR", confidence=0.0,
                    make="", model="", year="",
                    error=str(exc),
                ))
        return results

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, paths, tfm):
            self.paths, self.tfm = paths, tfm
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]
            try:
                img = Image.open(p).convert("RGB")
                return self.tfm(img), str(p), True
            except Exception:
                return torch.zeros(3, bundle.image_size, bundle.image_size), str(p), False

    def _collate(batch):
        ts, ps, oks = zip(*batch, strict=False)
        return torch.stack(ts), list(ps), list(oks)

    tfm    = _build_transform(bundle.image_size)
    loader = DataLoader(
        _DS(image_paths, tfm), batch_size=batch_size,
        shuffle=False, collate_fn=_collate,
        pin_memory=(bundle.device == "cuda"),
    )
    device = torch.device(bundle.device)
    results: list[PredictionResult] = []

    with torch.inference_mode():
        for tensors, paths, oks in loader:
            tensors = tensors.to(device)
            logits  = bundle.model(tensors)
            probs   = F.softmax(logits, dim=1)
            k       = min(top_k, len(bundle.class_names))
            topk_p, topk_i = probs.topk(k, dim=1)

            for idx, (path, ok) in enumerate(zip(paths, oks, strict=False)):
                if not ok:
                    results.append(PredictionResult(
                        filename=Path(path).name, filepath=path,
                        prediction="ERROR", confidence=0.0,
                        make="", model="", year="",
                        error="Could not read image",
                    ))
                    continue

                top_idx = topk_i[idx, 0].item()
                label = bundle.class_names[top_idx]
                conf  = topk_p[idx, 0].item()
                make, car_model, year = _resolve_fields(top_idx, label, bundle)
                entries = [
                    TopKEntry(bundle.class_names[topk_i[idx, j].item()], topk_p[idx, j].item())
                    for j in range(k)
                ]
                results.append(PredictionResult(
                    filename=Path(path).name, filepath=path,
                    prediction=label, confidence=conf,
                    make=make, model=car_model, year=year,
                    top_k=entries,
                ))

    return results
