"""
Training script for CBAM / ResNet-50 experiments on CIFAR-100.

Usage examples:
  python train.py --model baseline
  python train.py --model cbam --reduction 16 --epochs 200
  python train.py --model channel --epochs 200 --save_dir checkpoints/
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn

from data.dataloader import get_cifar100_loaders
from models.resnet import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-50 variants on CIFAR-100')
    parser.add_argument('--model',       type=str,   default='baseline',
                        choices=['baseline', 'channel', 'spatial', 'cbam'],
                        help='Attention variant to train.')
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=0.1)
    parser.add_argument('--momentum',    type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=5e-4)
    parser.add_argument('--reduction',   type=int,   default=16,
                        help='Channel attention reduction ratio.')
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints')
    parser.add_argument('--data_root',   type=str,   default='./data')
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Path to checkpoint to resume from.')
    parser.add_argument('--num_classes', type=int,   default=100)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, labels, topk=(1,))[0]
        total_loss += loss.item()
        total_acc  += acc

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = len(loader)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        acc = accuracy(outputs, labels, topk=(1,))[0]
        total_loss += loss.item()
        total_acc  += acc

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    model = build_model(args.model, num_classes=args.num_classes, reduction=args.reduction)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: ResNet-50 [{args.model}]  |  Parameters: {n_params:,}')

    # Optimizer & scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Resume
    start_epoch = 0
    best_acc    = 0.0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer)
        print(f'Resumed from {args.resume}  (epoch {start_epoch}, best_acc {best_acc:.2f}%)')
        # Advance scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()

    # CSV log
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f'train_log_{args.model}_r{args.reduction}.csv')
    log_file = open(log_path, 'w', newline='')
    writer   = csv.writer(log_file)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr'])

    print(f'\nStarting training for {args.epochs} epochs...\n')
    print(f'{"Epoch":>6}  {"Train Loss":>10}  {"Train Acc":>9}  {"Test Loss":>9}  {"Test Acc":>8}  {"LR":>8}')
    print('-' * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(f'{epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>8.2f}%  '
              f'{test_loss:>9.4f}  {test_acc:>7.2f}%  {current_lr:>8.6f}  '
              f'({elapsed:.0f}s)')

        writer.writerow([epoch + 1, f'{train_loss:.4f}', f'{train_acc:.2f}',
                         f'{test_loss:.4f}', f'{test_acc:.2f}', f'{current_lr:.6f}'])
        log_file.flush()

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(
                args.save_dir, f'best_{args.model}_r{args.reduction}.pth'
            )
            save_checkpoint({
                'epoch':                epoch + 1,
                'model':                args.model,
                'reduction':            args.reduction,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc':             best_acc,
            }, ckpt_path)

    log_file.close()
    print(f'\nTraining complete. Best test accuracy: {best_acc:.2f}%')
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    main()
