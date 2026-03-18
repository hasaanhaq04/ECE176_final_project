"""
Ablation runner — trains all experimental variants and prints a summary table.

Experiment 2: Four attention variants
  baseline, channel, spatial, cbam  (each for --epochs epochs)

Experiment 4: Reduction ratio sweep
  cbam with r in {8, 16, 32}

Run everything:
  python ablation.py --epochs 200

Dry-run (smoke test):
  python ablation.py --epochs 5

After training, a summary table is printed and saved to results/ablation_summary.csv.
"""

import argparse
import csv
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Run all ablation experiments')
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints')
    parser.add_argument('--data_root',   type=str,   default='./data')
    parser.add_argument('--results_dir', type=str,   default='results')
    # Skip already-trained variants (useful for resuming)
    parser.add_argument('--skip',        type=str,   nargs='*', default=[],
                        help='Variant names to skip, e.g. --skip baseline channel')
    return parser.parse_args()


def run_training(model: str, reduction: int, args) -> float:
    """Launch train.py as a subprocess and return the best test accuracy."""
    cmd = [
        sys.executable, 'train.py',
        '--model',        model,
        '--reduction',    str(reduction),
        '--epochs',       str(args.epochs),
        '--batch_size',   str(args.batch_size),
        '--num_workers',  str(args.num_workers),
        '--save_dir',     args.save_dir,
        '--data_root',    args.data_root,
    ]
    print(f'\n{"="*60}')
    print(f'  Training: model={model}, reduction={reduction}')
    print(f'{"="*60}')

    result = subprocess.run(cmd, check=True)

    # Read best accuracy from the log CSV
    log_path = os.path.join(args.save_dir, f'train_log_{model}_r{reduction}.csv')
    best_acc = _read_best_acc(log_path)
    return best_acc


def _read_best_acc(log_path: str) -> float:
    """Parse training log CSV and return the highest test_acc value."""
    if not os.path.isfile(log_path):
        return float('nan')
    best = 0.0
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                acc = float(row['test_acc'])
                if acc > best:
                    best = acc
            except (ValueError, KeyError):
                pass
    return best


def count_params(model_name: str, reduction: int) -> int:
    """Instantiate the model and count parameters."""
    import torch
    from models.resnet import build_model
    model = build_model(model_name, num_classes=100, reduction=reduction)
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Experiment 2: all four attention variants (reduction=16 throughout)
    # -----------------------------------------------------------------------
    exp2_variants = ['baseline', 'channel', 'spatial', 'cbam']
    exp2_results  = []

    for variant in exp2_variants:
        tag = f'{variant}_r16'
        if tag in args.skip:
            print(f'Skipping {tag}')
            log_path = os.path.join(args.save_dir, f'train_log_{variant}_r16.csv')
            best_acc = _read_best_acc(log_path)
        else:
            best_acc = run_training(variant, reduction=16, args=args)
        n_params = count_params(variant, reduction=16)
        exp2_results.append({'variant': variant, 'reduction': 16,
                             'best_acc': best_acc, 'n_params': n_params})

    # -----------------------------------------------------------------------
    # Experiment 4: reduction ratio sweep on cbam
    # -----------------------------------------------------------------------
    exp4_results = []
    for r in [8, 16, 32]:
        tag = f'cbam_r{r}'
        if tag in args.skip or r == 16:
            # r=16 was already trained above
            log_path = os.path.join(args.save_dir, f'train_log_cbam_r{r}.csv')
            best_acc = _read_best_acc(log_path)
        else:
            best_acc = run_training('cbam', reduction=r, args=args)
        n_params = count_params('cbam', reduction=r)
        exp4_results.append({'variant': f'cbam_r{r}', 'reduction': r,
                             'best_acc': best_acc, 'n_params': n_params})

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_results = exp2_results + [r for r in exp4_results if r['reduction'] != 16]

    print('\n' + '='*65)
    print(f'{"Variant":<20} {"Reduction":>9} {"Best Acc (%)":>12} {"Params":>12}')
    print('-'*65)
    for r in all_results:
        print(f'{r["variant"]:<20} {r["reduction"]:>9} {r["best_acc"]:>11.2f}% {r["n_params"]:>12,}')
    print('='*65)

    summary_path = os.path.join(args.results_dir, 'ablation_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'reduction', 'best_acc', 'n_params'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\nSummary saved to: {summary_path}')


if __name__ == '__main__':
    main()
