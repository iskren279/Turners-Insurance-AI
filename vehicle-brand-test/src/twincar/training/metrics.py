"""Classification metrics for the flat baseline.

Top-1/3/5 accuracy on ``fine_class`` is computed directly from the logits. Make /
model / year accuracy is *derived*: the predicted (top-1) fine class is mapped to
its make / model / year via :class:`twincar.data.label_maps.LabelMaps` and
compared against the ground-truth fine class mapped the same way. This gives the
auxiliary metrics for free, before any dedicated heads exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from twincar.data.label_maps import LabelMaps


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...]) -> dict[int, int]:
    """Return, per ``k``, the count of samples whose target is in the top-``k``.

    ``k`` values larger than the number of classes are clamped to the class count.
    """
    num_classes = logits.size(1)
    maxk = min(max(ks), num_classes)
    _, pred = logits.topk(maxk, dim=1)  # (batch, maxk)
    hits = pred.eq(targets.view(-1, 1))  # (batch, maxk)
    out: dict[int, int] = {}
    for k in ks:
        kk = min(k, num_classes)
        out[k] = int(hits[:, :kk].any(dim=1).sum().item())
    return out


@dataclass
class ClassificationMeter:
    """Accumulate counts across val batches and emit a metrics dict.

    Make / model / year correctness is derived from the top-1 fine prediction via
    the ``fine_to_*`` maps held on :class:`LabelMaps`.
    """

    label_maps: LabelMaps
    ks: tuple[int, ...] = (1, 3, 5)
    total: int = 0
    topk_hits: dict[int, int] = field(default_factory=dict)
    make_hits: int = 0
    model_hits: int = 0
    year_hits: int = 0
    loss_sum: float = 0.0

    def __post_init__(self) -> None:
        self.topk_hits = {k: 0 for k in self.ks}
        self._fine_to_make = torch.tensor(self.label_maps.fine_to_make, dtype=torch.long)
        self._fine_to_model = torch.tensor(self.label_maps.fine_to_model, dtype=torch.long)
        self._fine_to_year = torch.tensor(self.label_maps.fine_to_year, dtype=torch.long)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        logits = logits.detach().cpu()
        targets = targets.detach().cpu()
        batch = targets.size(0)
        self.total += batch
        self.loss_sum += loss * batch

        for k, hits in topk_correct(logits, targets, self.ks).items():
            self.topk_hits[k] += hits

        pred_fine = logits.argmax(dim=1)
        self.make_hits += int(
            (self._fine_to_make[pred_fine] == self._fine_to_make[targets]).sum().item()
        )
        self.model_hits += int(
            (self._fine_to_model[pred_fine] == self._fine_to_model[targets]).sum().item()
        )
        self.year_hits += int(
            (self._fine_to_year[pred_fine] == self._fine_to_year[targets]).sum().item()
        )

    def compute(self, prefix: str = "val") -> dict[str, float]:
        """Return a ``{prefix}/...`` metrics dict; zero counts give 0.0."""
        n = max(self.total, 1)
        metrics: dict[str, float] = {
            f"{prefix}/loss": self.loss_sum / n,
            f"{prefix}/fine_acc": self.topk_hits.get(1, 0) / n,
            f"{prefix}/make_acc": self.make_hits / n,
            f"{prefix}/model_acc": self.model_hits / n,
            f"{prefix}/year_acc": self.year_hits / n,
        }
        for k in self.ks:
            if k != 1:
                metrics[f"{prefix}/top{k}_acc"] = self.topk_hits.get(k, 0) / n
        return metrics
