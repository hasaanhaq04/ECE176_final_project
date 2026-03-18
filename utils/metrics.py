"""
Training metrics: top-k accuracy and confusion matrix accumulator.
"""

import torch
import numpy as np


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """
    Compute top-k accuracy for each k in topk.

    Args:
        output: Raw logits of shape (N, C).
        target: Ground-truth class indices of shape (N,).
        topk:   Tuple of k values to evaluate.

    Returns:
        List of float percentages, one per k.
    """
    with torch.no_grad():
        maxk    = max(topk)
        batch_n = target.size(0)

        # (N, maxk) — indices of top-k predictions per sample
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()                              # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N)

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k * 100.0 / batch_n).item())
        return results


class ConfusionMatrix:
    """
    Accumulates a confusion matrix over multiple batches.

    Args:
        num_classes: Number of classes (100 for CIFAR-100).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """
        Add predictions from one batch.

        Args:
            output: Logits (N, C) or predicted class indices (N,).
            target: Ground-truth class indices (N,).
        """
        if output.dim() == 2:
            preds = output.argmax(dim=1)
        else:
            preds = output

        preds  = preds.cpu().numpy()
        target = target.cpu().numpy()

        for p, t in zip(preds, target):
            self.matrix[t, p] += 1

    def reset(self) -> None:
        self.matrix.fill(0)

    def get_matrix(self) -> np.ndarray:
        return self.matrix.copy()

    def per_class_accuracy(self) -> np.ndarray:
        """Returns per-class accuracy as a 1-D array."""
        diag  = np.diag(self.matrix).astype(float)
        total = self.matrix.sum(axis=1).astype(float)
        total = np.where(total == 0, 1, total)   # avoid div-by-zero
        return diag / total
