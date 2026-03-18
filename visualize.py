"""
Attention map visualization for trained CBAM / spatial-attention models.

For each sampled image the spatial attention maps from every SpatialAttention
module in the network are captured via forward hooks, upsampled to 32×32, and
overlaid as a heatmap on the original image.

Also produces a side-by-side figure comparing attention on correctly classified
vs. misclassified examples.

Outputs (saved to results/):
  results/attention_maps.png          — grid of input + overlaid attention
  results/attention_correct_vs_wrong.png

Usage:
  python visualize.py --checkpoint checkpoints/best_cbam_r16.pth --num_images 16
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.resnet import build_model
from utils.checkpoint import load_checkpoint
from data.dataloader import CIFAR100_MEAN, CIFAR100_STD


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize spatial attention maps')
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--model',       type=str, default='cbam',
                        choices=['baseline', 'channel', 'spatial', 'cbam'])
    parser.add_argument('--reduction',   type=int, default=16)
    parser.add_argument('--num_images',  type=int, default=16,
                        help='Number of images to visualize.')
    parser.add_argument('--data_root',   type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=100)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hook-based attention capture
# ---------------------------------------------------------------------------

class AttentionHook:
    """Captures the output of SpatialAttention sigmoid operations via forward hook."""

    def __init__(self):
        self.maps: list[torch.Tensor] = []
        self._handles = []

    def register(self, model: torch.nn.Module) -> None:
        from models.cbam import SpatialAttention
        for module in model.modules():
            if isinstance(module, SpatialAttention):
                handle = module.register_forward_hook(self._hook_fn)
                self._handles.append(handle)

    def _hook_fn(self, module, input, output):
        # output shape: (B, C, H, W) — the spatially re-weighted feature map
        # We need the attention weights, which are the ratio output/input
        # Instead, patch: re-run sigmoid(conv(cat(avg,max))) manually.
        # Simpler: use the stored input and output to recover the map.
        inp = input[0]                                       # (B, C, H, W) before spatial scaling
        # attention = output / (inp + 1e-8)  gives the map per channel (all same)
        with torch.no_grad():
            avg_out = inp.mean(dim=1, keepdim=True)
            max_out = inp.amax(dim=1, keepdim=True)
            combined = torch.cat([avg_out, max_out], dim=1)
            attn_map = torch.sigmoid(module.conv(combined))  # (B, 1, H, W)
        self.maps.append(attn_map.detach().cpu())

    def clear(self):
        self.maps.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to an HWC uint8 numpy image."""
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    img  = img_tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_attention(img_np: np.ndarray, attn_map: torch.Tensor) -> np.ndarray:
    """
    Overlay a spatial attention map on an RGB image.

    attn_map: (1, H, W) tensor with values in [0, 1].
    Returns an HWC uint8 image.
    """
    H, W = img_np.shape[:2]
    # Upsample attention map to image size
    attn = F.interpolate(
        attn_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Normalise to [0, 1]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Heatmap via matplotlib colormap
    cmap    = plt.get_cmap('jet')
    heatmap = (cmap(attn)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    blended = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
    return blended


def get_last_attn_map(hook: AttentionHook) -> torch.Tensor | None:
    """Return the final-layer attention map from the hook list (one image)."""
    if not hook.maps:
        return None
    # Last map corresponds to the deepest SpatialAttention layer
    return hook.maps[-1][0]  # (1, H, W)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_attention_grid(records: list[dict], save_path: str, cols: int = 4) -> None:
    """
    Plot a grid of (original image, attention overlay) pairs.

    Each record: {'img': np.ndarray HWC, 'overlay': np.ndarray HWC,
                  'true': int, 'pred': int, 'correct': bool}
    """
    n    = len(records)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
    axes = np.array(axes).reshape(rows * 2, cols)

    for i, rec in enumerate(records):
        row, col = divmod(i, cols)
        r_img = row * 2
        r_attn = r_img + 1

        axes[r_img,  col].imshow(rec['img'])
        axes[r_img,  col].axis('off')
        color = 'green' if rec['correct'] else 'red'
        axes[r_img,  col].set_title(
            f"True:{rec['true']} Pred:{rec['pred']}", fontsize=7, color=color
        )

        if rec['overlay'] is not None:
            axes[r_attn, col].imshow(rec['overlay'])
        else:
            axes[r_attn, col].imshow(rec['img'])
        axes[r_attn, col].axis('off')
        axes[r_attn, col].set_title('Attn', fontsize=7)

    # Hide unused axes
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row * 2,     col].axis('off')
        axes[row * 2 + 1, col].axis('off')

    plt.suptitle('Spatial Attention Maps (bottom row) vs. Input (top row)', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Attention grid saved to: {save_path}')


def plot_correct_vs_wrong(correct_records: list[dict], wrong_records: list[dict],
                          save_path: str, n: int = 4) -> None:
    """
    Side-by-side comparison: n correctly classified vs n misclassified examples.
    Each column shows (original, attention overlay).
    """
    correct_records = correct_records[:n]
    wrong_records   = wrong_records[:n]
    total = max(len(correct_records), len(wrong_records))
    if total == 0:
        return

    fig, axes = plt.subplots(4, total * 2, figsize=(total * 6, 10))

    def fill(records, col_offset, label):
        for i, rec in enumerate(records):
            col = col_offset + i * 2
            # Column header on first image
            axes[0, col].imshow(rec['img'])
            axes[0, col].axis('off')
            axes[0, col].set_title(f'{label}\nTrue:{rec["true"]} Pred:{rec["pred"]}',
                                   fontsize=7)
            axes[1, col].axis('off')  # spacer

            if rec['overlay'] is not None:
                axes[2, col].imshow(rec['overlay'])
            else:
                axes[2, col].imshow(rec['img'])
            axes[2, col].axis('off')
            axes[2, col].set_title('Attention', fontsize=7)
            axes[3, col].axis('off')

    fill(correct_records, 0,       'Correct')
    fill(wrong_records,   total,   'Wrong')

    plt.suptitle('Correct (left) vs Misclassified (right) — Attention Comparison', fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Correct vs wrong attention figure saved to: {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    model = build_model(args.model, num_classes=args.num_classes, reduction=args.reduction)
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    # Register attention hooks
    hook = AttentionHook()
    hook.register(model)

    has_spatial = bool(hook._handles)
    if not has_spatial:
        print(f'[WARNING] Model "{args.model}" has no SpatialAttention modules. '
              'Overlays will show original images only.')

    # Dataset (no augmentation for visualization)
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    test_dataset = datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=norm
    )

    # Random sample
    indices = random.sample(range(len(test_dataset)), args.num_images)
    records        = []
    correct_records = []
    wrong_records   = []

    with torch.no_grad():
        for idx in indices:
            img_tensor, true_label = test_dataset[idx]
            hook.clear()

            inp    = img_tensor.unsqueeze(0).to(device)
            output = model(inp)
            pred   = output.argmax(dim=1).item()
            correct = (pred == true_label)

            img_np  = denormalize(img_tensor)
            attn_map = get_last_attn_map(hook)
            overlay  = overlay_attention(img_np, attn_map) if attn_map is not None else None

            rec = {'img': img_np, 'overlay': overlay,
                   'true': true_label, 'pred': pred, 'correct': correct}
            records.append(rec)
            (correct_records if correct else wrong_records).append(rec)

    hook.remove()

    # Save figures
    grid_path = os.path.join(args.results_dir, f'attention_maps_{args.model}_r{args.reduction}.png')
    plot_attention_grid(records, grid_path)

    cw_path = os.path.join(args.results_dir,
                           f'attention_correct_vs_wrong_{args.model}_r{args.reduction}.png')
    plot_correct_vs_wrong(correct_records, wrong_records, cw_path)


if __name__ == '__main__':
    main()
