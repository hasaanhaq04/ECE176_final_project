"""
Evaluate a trained checkpoint on the CIFAR-100 test set.

Outputs:
  - Top-1 test accuracy
  - Parameter count
  - Confusion matrix saved to results/confusion_<model>_r<reduction>.png

Usage:
  python evaluate.py --model cbam --checkpoint checkpoints/best_cbam_r16.pth
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataloader import get_cifar100_loaders
from models.resnet import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import accuracy, ConfusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on CIFAR-100')
    parser.add_argument('--model',      type=str, required=True,
                        choices=['baseline', 'channel', 'spatial', 'cbam'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the .pth checkpoint file.')
    parser.add_argument('--reduction',  type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--data_root',  type=str, default='./data')
    parser.add_argument('--results_dir',type=str, default='results')
    parser.add_argument('--num_classes',type=int, default=100)
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device, num_classes):
    model.eval()
    cm          = ConfusionMatrix(num_classes)
    total_acc   = 0.0
    n_batches   = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        acc     = accuracy(outputs, labels, topk=(1,))[0]
        total_acc += acc
        cm.update(outputs, labels)

    return total_acc / n_batches, cm


def plot_confusion_matrix(matrix: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix — CIFAR-100', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to: {save_path}')


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    _, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        pin_memory=(device.type == 'cuda'),
    )

    model = build_model(args.model, num_classes=args.num_classes, reduction=args.reduction)
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: ResNet-50 [{args.model}]  |  Parameters: {n_params:,}')

    test_acc, cm = run_evaluation(model, test_loader, device, args.num_classes)
    print(f'Top-1 Test Accuracy: {test_acc:.2f}%')

    # Per-class accuracy summary
    per_class = cm.per_class_accuracy()
    print(f'Mean per-class accuracy: {per_class.mean()*100:.2f}%')
    print(f'Min per-class accuracy:  {per_class.min()*100:.2f}%  '
          f'(class {per_class.argmin()})')
    print(f'Max per-class accuracy:  {per_class.max()*100:.2f}%  '
          f'(class {per_class.argmax()})')

    # Save confusion matrix
    cm_path = os.path.join(
        args.results_dir, f'confusion_{args.model}_r{args.reduction}.png'
    )
    plot_confusion_matrix(cm.get_matrix(), cm_path)


if __name__ == '__main__':
    main()
