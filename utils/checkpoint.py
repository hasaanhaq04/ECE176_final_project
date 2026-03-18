"""
Checkpoint save / load utilities.
"""

import os
import torch


def save_checkpoint(state: dict, filepath: str) -> None:
    """
    Save a training checkpoint.

    Args:
        state:    Dictionary containing at minimum:
                    'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc'
        filepath: Full path to write the .pth file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        filepath:  Path to the saved .pth file.
        model:     Model whose weights will be restored.
        optimizer: If provided, restore optimizer state as well.

    Returns:
        (epoch, best_acc) — the epoch and best accuracy stored in the checkpoint.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch    = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    return epoch, best_acc
