"""Binary classification and threshold metrics for offline evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _arrays(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores and labels must be non-empty and have equal length.")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("scores must be finite and labels must be binary.")
    return scores, labels


def confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> list[list[int]]:
    scores, labels = _arrays(scores, labels)
    predictions = scores >= threshold
    tn = int(((~predictions) & (labels == 0)).sum())
    fp = int((predictions & (labels == 0)).sum())
    fn = int(((~predictions) & (labels == 1)).sum())
    tp = int((predictions & (labels == 1)).sum())
    return [[tn, fp], [fn, tp]]


def _eer(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores, labels = _arrays(scores, labels)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    far = np.cumsum(sorted_labels == 0) / negatives
    frr = (positives - np.cumsum(sorted_labels == 1)) / positives
    crossing = int(np.argmin(np.abs(far - frr)))
    return float((far[crossing] + frr[crossing]) / 2.0)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores, labels = _arrays(scores, labels)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return None
    ranks = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort") + 1
    rank_sum = ranks[positives].sum()
    return float((rank_sum - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))


def classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    scores, labels = _arrays(scores, labels)
    matrix = confusion_matrix(scores, labels, threshold)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": _eer(scores, labels),
        "roc_auc": roc_auc(scores, labels),
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "frr": fn / (fn + tp) if fn + tp else 0.0,
        "confusion_matrix": matrix,
    }


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Select the threshold maximizing validation F1; ties prefer 0.5 proximity."""
    scores, labels = _arrays(scores, labels)
    candidates = np.unique(np.concatenate((np.array([0.0, 0.5, 1.0]), scores)))
    ranked = sorted(
        candidates,
        key=lambda threshold: (
            classification_metrics(scores, labels, float(threshold))["f1"],
            -abs(float(threshold) - 0.5),
        ),
        reverse=True,
    )
    return float(ranked[0])


__all__ = ["classification_metrics", "confusion_matrix", "roc_auc", "select_threshold"]