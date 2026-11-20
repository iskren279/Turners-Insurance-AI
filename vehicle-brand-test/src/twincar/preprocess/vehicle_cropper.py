"""
src/twincar/preprocess/vehicle_cropper.py — optional YOLO vehicle cropper

EXPERIMENTAL · OPT-IN. This module is **not** wired into the default
inference, API, or Gradio paths. It exists to A/B test whether detecting and
cropping the vehicle before classification helps on surveillance-like imagery
(motivated by the CompCars reconnaissance — occlusion, varied angles, small
vehicles in frame). Default inference stays full-image until the A/B evidence
says otherwise.

Design rules (see docs/experiments/yolo-cropper-ab-v1.md):
  * Pretrained YOLO only — never trained here. Defaults to ultralytics
    ``yolov8n.pt`` (swap to ``yolo11n.pt`` via ``weights=`` if preferred).
  * ``ultralytics`` is an *optional* dependency (``pip install -e ".[detect]"``).
    The import is lazy and isolated in :func:`_load_yolo`, so importing this
    module — and therefore the whole package, CI, and stub mode — never needs
    ultralytics and never downloads weights.
  * Any failure (ultralytics missing, no vehicle found, low confidence, decode
    or inference error) returns the **full image unchanged** with
    ``detected=False`` and a ``reason``. Callers can always fall back safely.

The detection-parsing and geometry helpers (:func:`parse_results`,
:func:`select_best_box`, :func:`apply_margin`) are pure and unit-tested with
synthetic boxes — no ultralytics, no torch, no weight downloads.

Usage::

    from twincar.preprocess.vehicle_cropper import VehicleCropper

    cropper = VehicleCropper(conf_threshold=0.25, margin=0.12)
    result = cropper.crop(pil_image)
    image_to_classify = result.image          # cropped, or full image on fallback
    if result.detected:
        print(result.class_name, result.confidence, result.box)
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_MARGIN = 0.12  # 12 %, within the 10–15 % band from the experiment brief
DEFAULT_DEVICE = "cpu"

# COCO vehicle categories. Matched against the model's own ``names`` mapping, so
# a non-COCO model that lacks these labels simply yields no candidates and the
# cropper falls back to the full image ("if supported by model labels").
DEFAULT_VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle")


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """One detected box in pixel coordinates. Pure data — no torch tensors."""

    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


@dataclass
class CropResult:
    """Outcome of a crop attempt.

    ``image`` is always usable: the margin-expanded crop when ``detected`` is
    True, otherwise the (RGB-converted) full input image. ``reason`` records why
    a fallback happened, which the A/B script surfaces for triage.
    """

    image: Image.Image
    detected: bool
    reason: str
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None
    class_name: str = ""


# ── Availability check ──────────────────────────────────────────────────────────


def yolo_available() -> bool:
    """True if ``ultralytics`` is importable. Lets tests/CLI skip cleanly.

    Uses ``find_spec`` so it never imports ultralytics (and never triggers a
    weight download) just to answer the question.
    """
    return importlib.util.find_spec("ultralytics") is not None


# ── Pure helpers (no ultralytics / no torch) ─────────────────────────────────────


def _as_list(value: Any) -> list:
    """Normalise a torch tensor or array-like to a plain Python list.

    Real ultralytics boxes expose ``.tolist()``; test fakes pass plain lists.
    """
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else list(value)


def _name_for(names: Any, class_id: int) -> str:
    """Resolve a class id to its label via the model's ``names`` mapping.

    ultralytics uses a ``{id: name}`` dict; tolerate list-like mappings too and
    fall back to the stringified id when the label is unknown.
    """
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    try:
        return names[class_id]
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def parse_results(results: Any, names: Any) -> list[Detection]:
    """Flatten ultralytics ``Results`` into a list of :class:`Detection`.

    Reads ``result.boxes.{xyxy,conf,cls}`` as array-likes, so it works with both
    real tensors and synthetic lists. Results without boxes are skipped.
    """
    detections: list[Detection] = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = _as_list(boxes.xyxy)
        confs = _as_list(boxes.conf)
        clss = _as_list(boxes.cls)
        for coords, conf, cls_id in zip(xyxy, confs, clss, strict=True):
            cid = int(cls_id)
            x1, y1, x2, y2 = (float(v) for v in coords)
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cid,
                    class_name=_name_for(names, cid),
                )
            )
    return detections


def select_best_box(
    detections: list[Detection],
    allowed_names: tuple[str, ...] | list[str],
    conf_threshold: float,
) -> Detection | None:
    """Pick the highest-confidence vehicle box at or above ``conf_threshold``.

    Filters to detections whose class is in ``allowed_names`` (case-insensitive).
    Returns None when nothing qualifies, signalling a full-image fallback.
    """
    allowed = {name.lower() for name in allowed_names}
    candidates = [
        d
        for d in detections
        if d.class_name.lower() in allowed and d.confidence >= conf_threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def apply_margin(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Expand ``box`` by ``margin`` (fraction of each side) and clamp to image.

    Returns an integer ``(x1, y1, x2, y2)`` crop box inside the image. If the
    result is degenerate (zero area), returns the full-image box so the caller
    still gets a valid crop.
    """
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin

    nx1 = max(0, int(round(x1 - dx)))
    ny1 = max(0, int(round(y1 - dy)))
    nx2 = min(width, int(round(x2 + dx)))
    ny2 = min(height, int(round(y2 + dy)))

    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, width, height
    return nx1, ny1, nx2, ny2


# ── Lazy model construction (the only ultralytics touch-point) ───────────────────


def _load_yolo(weights: str) -> Any:
    """Import ultralytics lazily and build a YOLO model.

    Isolated here so the whole pipeline stays import-clean without ultralytics,
    and so tests can monkeypatch this one function to inject a fake model (or
    simulate ultralytics being absent) without ever downloading weights.
    """
    from ultralytics import YOLO

    return YOLO(weights)


# ── Cropper ──────────────────────────────────────────────────────────────────


class VehicleCropper:
    """Detect the most confident vehicle and return a margin-expanded crop.

    The model is loaded lazily on the first :meth:`crop` call and cached. Every
    failure mode degrades to returning the full image, so a cropper instance is
    always safe to call regardless of whether ultralytics is installed.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        vehicle_classes: tuple[str, ...] = DEFAULT_VEHICLE_CLASSES,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.margin = margin
        self.vehicle_classes = tuple(vehicle_classes)
        self.device = device
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_yolo(self.weights)
        return self._model

    def crop(self, image: Image.Image) -> CropResult:
        """Return a :class:`CropResult` for ``image`` (never raises).

        Loads the model on first use, runs detection, selects the best vehicle
        box, expands by the margin, and crops. Any problem — ultralytics
        missing, inference error, no vehicle, low confidence — returns the full
        image with ``detected=False`` and an explanatory ``reason``.
        """
        rgb = image.convert("RGB")
        width, height = rgb.size

        try:
            model = self._ensure_model()
        except Exception as exc:  # ImportError or weight-load failure
            log.warning("YOLO unavailable (%s) — using full image.", exc)
            return CropResult(image=rgb, detected=False, reason=f"yolo-unavailable: {exc}")

        try:
            results = model.predict(
                rgb,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )
            detections = parse_results(results, getattr(model, "names", {}))
        except Exception as exc:
            log.warning("YOLO detection failed (%s) — using full image.", exc)
            return CropResult(image=rgb, detected=False, reason=f"detection-error: {exc}")

        best = select_best_box(detections, self.vehicle_classes, self.conf_threshold)
        if best is None:
            return CropResult(image=rgb, detected=False, reason="no-vehicle-detected")

        crop_box = apply_margin(best.box, width, height, self.margin)
        cropped = rgb.crop(crop_box)
        return CropResult(
            image=cropped,
            detected=True,
            reason="ok",
            confidence=best.confidence,
            box=crop_box,
            class_name=best.class_name,
        )
