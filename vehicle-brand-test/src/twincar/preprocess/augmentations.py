"""Car-specific augmentations and robust-evaluation transforms.

This module is deliberately separate from :mod:`twincar.data.dataset` so that
the training transform used by the baseline stays untouched. It provides:

* :class:`RandomShadow` — a lightweight cast-shadow overlay (torchvision has no
  native shadow transform), driven by the global ``torch`` RNG so a single
  :func:`torch.manual_seed` makes a whole pipeline reproducible.
* :func:`train_augmentation_v1` — the documented "augmentation v1" stack used by
  ``scripts/visualize_augmentations.py``. See ``docs/experiments/augmentation-v1.md``.
* :func:`robust_light` / :func:`robust_hard` — *evaluation-only* transforms that
  simulate production-like drone / robot inspection conditions. ``robust_light``
  is a realistic mild shift; ``robust_hard`` is a stronger diagnostic stress test
  and is **not** meant for training/checkpoint selection.

All eval transforms end in ``ToTensor`` + ImageNet ``Normalize`` so their output
is drop-in compatible with the trained model (``(3, image_size, image_size)``).
"""

from __future__ import annotations

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision import transforms

# Single source of truth for the normalisation the pretrained backbone expects.
from twincar.data.dataset import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "RandomShadow",
    "train_augmentation_v1",
    "train_augmentation_v2",
    "robust_light",
    "robust_hard",
    "robust_occlusion",
    "build_robust_transform",
    "ROBUST_PRESETS",
]


class RandomShadow:
    """Overlay a soft, randomly placed dark patch to mimic a cast shadow.

    With probability ``p`` a rectangular region (a random fraction of the image)
    is darkened by a random factor and its mask is Gaussian-blurred for soft
    edges. Operates on and returns a ``PIL.Image`` so it can sit anywhere in a
    torchvision ``Compose`` before ``ToTensor``. Randomness uses the global
    ``torch`` RNG, so seeding ``torch`` alone makes it deterministic.

    Args:
        p: Probability of applying a shadow at all.
        frac: ``(min, max)`` fraction of each side covered by the shadow patch.
        intensity: ``(min, max)`` darkening factor; ``1.0`` is unchanged, lower
            is darker (passed to :class:`PIL.ImageEnhance.Brightness`).
    """

    def __init__(
        self,
        p: float = 0.3,
        frac: tuple[float, float] = (0.3, 0.6),
        intensity: tuple[float, float] = (0.4, 0.7),
    ) -> None:
        self.p = p
        self.frac = frac
        self.intensity = intensity

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand(()))

    def __call__(self, img: Image.Image) -> Image.Image:
        if float(torch.rand(())) >= self.p:
            return img

        w, h = img.size
        rect_w = max(1, int(w * self._uniform(*self.frac)))
        rect_h = max(1, int(h * self._uniform(*self.frac)))
        x0 = int(self._uniform(0.0, max(0, w - rect_w)))
        y0 = int(self._uniform(0.0, max(0, h - rect_h)))

        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.05))

        factor = self._uniform(*self.intensity)
        darkened = ImageEnhance.Brightness(img).enhance(factor)
        return Image.composite(darkened, img, mask)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(p={self.p}, frac={self.frac}, intensity={self.intensity})"


def _normalize_tail() -> list:
    """The shared tensor tail: PIL -> normalised CHW tensor."""
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def train_augmentation_v1(image_size: int, *, to_tensor: bool = True) -> transforms.Compose:
    """Build the car-specific "augmentation v1" training stack.

    Order mirrors the documentation: geometric framing first, then photometric
    and degradation effects, then normalisation. Set ``to_tensor=False`` to get a
    PIL-only pipeline (used by the visualisation script so tiles can be pasted
    into a grid without denormalising).
    """
    pil_stage = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        RandomShadow(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ]
    if not to_tensor:
        return transforms.Compose(pil_stage)
    return transforms.Compose([*pil_stage, *_normalize_tail()])


def train_augmentation_v2(image_size: int) -> transforms.Compose:
    """Augmentation v1 **plus occlusion** — the one production condition v1 omits.

    The brief lists partial occlusion (≤10–40%) as a production input, but v1 has
    no occlusion op. ``RandomErasing`` blanks a random region to mimic it. It acts
    on tensors, so it must follow ``ToTensor``; appending it to the full v1 stack
    keeps v1 as the single source of the geometry/photometric steps (DRY) and adds
    occlusion last. The eval-only :func:`robust_occlusion` mirrors this so the
    gain is actually measurable.
    """
    base = train_augmentation_v1(image_size, to_tensor=True)
    return transforms.Compose(
        [
            *base.transforms,
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ]
    )


def robust_light(image_size: int) -> transforms.Compose:
    """Realistic mild field shift for robustness evaluation (eval-only).

    Centre-crop geometry like the clean eval transform, plus gentle exposure
    jitter, occasional soft blur, and a small perspective tilt — the kind of
    variation a well-positioned drone / robot camera still produces.

    Note: contains stochastic ops (``ColorJitter``, ``RandomApply`` + blur,
    ``RandomPerspective``). For reproducible metrics, call
    :func:`torch.manual_seed` before evaluation.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.14))),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            *_normalize_tail(),
        ]
    )


def robust_hard(image_size: int) -> transforms.Compose:
    """Strong diagnostic stress test for robustness evaluation (eval-only).

    Heavier exposure swings, frequent cast shadows, stronger blur, larger
    perspective tilt and translation. Intended to *measure* degradation under
    adverse conditions — do **not** use it for training or checkpoint selection.

    Note: every step here is stochastic. For reproducible metrics, call
    :func:`torch.manual_seed` before evaluation.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.14))),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            RandomShadow(p=0.6, frac=(0.4, 0.8), intensity=(0.25, 0.55)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0))], p=0.8),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            *_normalize_tail(),
        ]
    )


def robust_occlusion(image_size: int) -> transforms.Compose:
    """Occlusion-only robustness probe (eval-only).

    Clean centre-crop geometry (identical to the clean eval transform) plus a
    **guaranteed** ``RandomErasing`` patch covering 10–40% of the frame — the
    brief's stated occlusion range. ``p=1.0`` so every image is occluded, making
    the resulting accuracy a direct read on occlusion robustness. Stochastic in
    *where* it erases, so seed before evaluating for reproducible numbers.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.14))),
            transforms.CenterCrop(image_size),
            *_normalize_tail(),
            transforms.RandomErasing(p=1.0, scale=(0.1, 0.4), ratio=(0.3, 3.3), value=0),
        ]
    )


# Name -> builder, so eval code / CLIs can select a preset by string.
ROBUST_PRESETS = {
    "robust_light": robust_light,
    "robust_hard": robust_hard,
    "robust_occlusion": robust_occlusion,
}


def build_robust_transform(name: str, image_size: int) -> transforms.Compose:
    """Return the robust eval transform registered under ``name``."""
    try:
        builder = ROBUST_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown robust preset '{name}'. Choose from {sorted(ROBUST_PRESETS)}."
        ) from None
    return builder(image_size)
