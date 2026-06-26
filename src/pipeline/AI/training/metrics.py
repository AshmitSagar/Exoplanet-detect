"""
metrics.py — Stateless metric utilities and a batch accumulator for the training loop.

Design notes
────────────
• All functions are pure / stateless — they operate on tensors or numpy arrays
  that are already gathered.  This makes them trivially testable in isolation.

• MetricAccumulator collects per-batch predictions/labels across an epoch and
  computes all metrics once at epoch-end rather than on every batch — this
  avoids the noise of per-batch F1/precision/recall on small batches.

• No sklearn dependency at inference time.  The implementations follow the
  standard confusion-matrix derivations directly.

Classes:
    MetricAccumulator  — accumulates preds/labels, computes epoch metrics

Functions:
    accuracy(preds, labels)                    → float
    precision_recall_f1(preds, labels, n_cls)  → dict[str, float]
    class_frequency(labels, n_cls)             → Tensor of shape (n_cls,)
"""

from __future__ import annotations

from typing import Dict, List

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Stateless utilities
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Top-1 accuracy.

    Args:
        preds  : Predicted class indices, shape (N,)  — argmax already applied.
        labels : Ground-truth class indices, shape (N,).

    Returns:
        Accuracy in [0.0, 1.0].
    """
    if preds.numel() == 0:
        return 0.0
    return (preds == labels).float().mean().item()


def precision_recall_f1(
    preds      : torch.Tensor,
    labels     : torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """
    Per-class and macro-averaged Precision, Recall, and F1.

    Computes from a confusion matrix; handles zero-denominator classes by
    returning 0.0 for that class (consistent with sklearn's zero_division=0).

    Args:
        preds       : Predicted class indices, shape (N,).
        labels      : Ground-truth class indices, shape (N,).
        num_classes : Total number of classes C.

    Returns:
        Dict with keys:
            "precision_macro", "recall_macro", "f1_macro"          — scalar floats
            "precision_cls_{i}", "recall_cls_{i}", "f1_cls_{i}"    — per-class floats
    """
    C = num_classes

    # Build confusion matrix on CPU
    p = preds.cpu().long()
    l = labels.cpu().long()
    conf = torch.zeros(C, C, dtype=torch.long)
    for pred_i, label_i in zip(p.tolist(), l.tolist()):
        conf[label_i, pred_i] += 1   # rows = true, cols = predicted

    # TP, FP, FN per class
    tp = conf.diag().float()                         # (C,)
    fp = conf.sum(dim=0).float() - tp               # predicted-col sum - TP
    fn = conf.sum(dim=1).float() - tp               # true-row sum - TP

    denom_p  = (tp + fp).clamp(min=1e-9)
    denom_r  = (tp + fn).clamp(min=1e-9)

    prec = tp / denom_p      # (C,)
    rec  = tp / denom_r      # (C,)

    denom_f1 = (prec + rec).clamp(min=1e-9)
    f1 = 2.0 * prec * rec / denom_f1   # (C,)

    result: Dict[str, float] = {}

    # Per-class
    for i in range(C):
        result[f"precision_cls_{i}"] = prec[i].item()
        result[f"recall_cls_{i}"]    = rec[i].item()
        result[f"f1_cls_{i}"]        = f1[i].item()

    # Macro averages
    result["precision_macro"] = prec.mean().item()
    result["recall_macro"]    = rec.mean().item()
    result["f1_macro"]        = f1.mean().item()

    return result


def class_frequency(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute the absolute frequency of each class in a label tensor.

    Useful for computing inverse-frequency class weights for CrossEntropyLoss.

    Args:
        labels      : Integer class labels, shape (N,).
        num_classes : Total number of classes C.

    Returns:
        Frequency tensor of shape (C,) with counts as floats.
    """
    freq = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        freq[c] = (labels == c).sum().float()
    return freq


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights: weight_c = total_samples / (C × freq_c).

    Classes with zero frequency get weight 0.0 (effectively ignored).
    This is the same formula used by sklearn's compute_class_weight('balanced').

    Args:
        labels      : Integer class labels from the *training* split, shape (N,).
        num_classes : Total number of classes C.

    Returns:
        Weight tensor of shape (C,); suitable to pass to nn.CrossEntropyLoss(weight=...).
    """
    freq = class_frequency(labels, num_classes)
    n    = labels.numel()
    # Avoid division by zero for absent classes
    weights = torch.where(
        freq > 0,
        torch.full_like(freq, n) / (num_classes * freq),
        torch.zeros_like(freq),
    )
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulates per-batch (loss, predictions, labels) across an epoch and
    computes all metrics at epoch end.

    Usage
    ─────
        acc = MetricAccumulator()
        for batch in loader:
            loss, logits, labels = ...
            acc.update(loss.item(), logits.argmax(1), labels)

        metrics = acc.compute(num_classes=4)
        # metrics["loss"], metrics["accuracy"], metrics["f1_macro"], ...

    Notes
    ─────
        • Predictions and labels are stored on CPU to avoid accumulating
          tensors on the GPU across many batches.
        • Loss is accumulated as a Python float (weighted by batch size) and
          divided by the total number of samples at the end.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total_loss   : float       = 0.0
        self._total_samples: int         = 0
        self._preds        : List[torch.Tensor] = []
        self._labels       : List[torch.Tensor] = []

    def update(
        self,
        loss   : float,
        preds  : torch.Tensor,   # (B,)  argmax already applied
        labels : torch.Tensor,   # (B,)
    ) -> None:
        """Accumulate one batch of results."""
        b = labels.numel()
        self._total_loss    += loss * b
        self._total_samples += b
        self._preds.append(preds.cpu())
        self._labels.append(labels.cpu())

    def compute(self, num_classes: int) -> Dict[str, float]:
        """
        Compute all metrics over accumulated batches.

        Returns
        ───────
        Dict with keys:
            "loss"             — mean cross-entropy loss
            "accuracy"         — top-1 accuracy
            "precision_macro"  — macro-averaged precision
            "recall_macro"     — macro-averaged recall
            "f1_macro"         — macro-averaged F1
            "precision_cls_i"  — per-class precision for i in range(num_classes)
            "recall_cls_i"     — per-class recall
            "f1_cls_i"         — per-class F1
        """
        if self._total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0,
                    "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

        all_preds  = torch.cat(self._preds,  dim=0)
        all_labels = torch.cat(self._labels, dim=0)

        result: Dict[str, float] = {}
        result["loss"]     = self._total_loss / self._total_samples
        result["accuracy"] = accuracy(all_preds, all_labels)
        result.update(precision_recall_f1(all_preds, all_labels, num_classes))
        return result
