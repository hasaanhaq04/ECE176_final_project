# ResNet-50 with CBAM Attention on CIFAR-100

ECE 176 Final Project — Investigating the effect of [CBAM (Convolutional Block Attention Module)](https://arxiv.org/abs/1807.06521) on image classification using CIFAR-100.

## Overview

This project compares a baseline ResNet-50 against variants augmented with different attention mechanisms:

| Variant | Description |
|---------|-------------|
| **Baseline** | Standard ResNet-50 (no attention) |
| **Channel** | Channel Attention only |
| **Spatial** | Spatial Attention only |
| **CBAM** | Full CBAM (Channel → Spatial) |

An additional ablation sweeps the CBAM reduction ratio across {8, 16, 32} to study its impact on capacity and accuracy.

## Results

| Variant | Reduction | Top-1 Accuracy | Parameters |
|---------|-----------|---------------|------------|
| Baseline | 16 | 79.13% | 23.7M |
| Channel | 16 | 80.09% | 26.2M |
| Spatial | 16 | 79.42% | 23.7M |
| **CBAM** | **16** | **80.73%** | **26.2M** |
| CBAM | 8 | 80.76% | 28.8M |
| CBAM | 32 | 79.97% | 25.0M |

## Project Structure

```
├── data/
│   ├── dataloader.py          # CIFAR-100 loaders with standard augmentation
│   └── __init__.py
├── models/
│   ├── cbam.py                # CBAM attention modules (Channel, Spatial, CBAM)
│   ├── resnet.py              # ResNet-50 backbone with pluggable attention
│   └── __init__.py
├── utils/
│   ├── checkpoint.py          # Save / load model checkpoints
│   ├── metrics.py             # Accuracy and confusion matrix helpers
│   └── __init__.py
├── train.py                   # Training script
├── evaluate.py                # Evaluation and confusion matrix generation
├── visualize.py               # Attention heatmap visualization
├── ablation.py                # Runs all ablation experiments end-to-end
└── test_unit.py               # Unit tests
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.4 compatible drivers (driver ≥ 520)

### Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib
```

## Usage

### Train a single variant

```bash
python train.py --model cbam --reduction 16 --epochs 200
```

Options for `--model`: `baseline`, `channel`, `spatial`, `cbam`

### Run full ablation study

```bash
python ablation.py --epochs 200
```

This trains all variants sequentially and writes a summary to `results/ablation_summary.csv`.

### Evaluate a checkpoint

```bash
python evaluate.py --model cbam --checkpoint checkpoints/best_cbam_r16.pth
```

### Visualize attention maps

```bash
python visualize.py --checkpoint checkpoints/best_cbam_r16.pth --num_images 16
```

## Training Details

- **Optimizer:** SGD (lr=0.1, momentum=0.9, weight decay=5e-4, Nesterov)
- **Scheduler:** Cosine Annealing over 200 epochs
- **Batch size:** 128
- **Data augmentation:** Random crop (32×32, padding=4), random horizontal flip
- **Architecture note:** CIFAR-adapted ResNet stem uses a 3×3 conv (stride 1) instead of the standard 7×7 + max-pool, since input images are 32×32.

## Reference

> Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* ECCV 2018.
