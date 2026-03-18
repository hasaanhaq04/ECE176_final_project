"""
Unit tests — run inside the project venv to verify all modules are correct.

  python test_unit.py

All assertions must pass before starting any training runs.
"""

import sys
import torch

# ---------------------------------------------------------------------------
# 0. Environment info
# ---------------------------------------------------------------------------
print(f'Python : {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    print(f'CUDA version   : {torch.version.cuda}')
print()

# ---------------------------------------------------------------------------
# 1. CBAM modules — shape preservation
# ---------------------------------------------------------------------------
print('=== Testing models/cbam.py ===')
from models.cbam import ChannelAttention, SpatialAttention, CBAM

x = torch.randn(2, 64, 8, 8)

ca  = ChannelAttention(64, reduction=16)
out = ca(x)
assert out.shape == x.shape, f'ChannelAttention shape mismatch: {out.shape}'
print(f'  ChannelAttention  {tuple(x.shape)} -> {tuple(out.shape)}   PASS')

sa  = SpatialAttention(kernel_size=7)
out = sa(x)
assert out.shape == x.shape, f'SpatialAttention shape mismatch: {out.shape}'
print(f'  SpatialAttention  {tuple(x.shape)} -> {tuple(out.shape)}   PASS')

cbam = CBAM(64, reduction=16, spatial_kernel=7)
out  = cbam(x)
assert out.shape == x.shape, f'CBAM shape mismatch: {out.shape}'
print(f'  CBAM              {tuple(x.shape)} -> {tuple(out.shape)}   PASS')

# Attention weights must be in (0, 1)
ca_weights = torch.sigmoid(ca.mlp(x.mean(dim=[2, 3])))
assert ca_weights.min() >= 0 and ca_weights.max() <= 1, 'Channel weights out of range'
print('  Channel attention weights in [0,1]               PASS')
print()

# ---------------------------------------------------------------------------
# 2. ResNet-50 variants — output shape and parameter counts
# ---------------------------------------------------------------------------
print('=== Testing models/resnet.py ===')
from models.resnet import build_model

dummy = torch.randn(2, 3, 32, 32)
param_counts = {}

for variant in ['baseline', 'channel', 'spatial', 'cbam']:
    model = build_model(variant, num_classes=100, reduction=16)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 100), f'{variant}: output shape {out.shape}'
    n = sum(p.numel() for p in model.parameters())
    param_counts[variant] = n
    print(f'  {variant:<10}  output {tuple(out.shape)}   params = {n:>12,}   PASS')

# CBAM must add parameters over baseline
assert param_counts['cbam']    > param_counts['baseline'], 'CBAM should have more params than baseline'
assert param_counts['channel'] > param_counts['baseline'], 'Channel should have more params than baseline'
assert param_counts['spatial'] > param_counts['baseline'], 'Spatial should have more params than baseline'
print()
print(f'  Param overhead (cbam - baseline): {param_counts["cbam"] - param_counts["baseline"]:,}')
print()

# ---------------------------------------------------------------------------
# 3. Data loader — batch shape check (downloads dataset if needed)
# ---------------------------------------------------------------------------
print('=== Testing data/dataloader.py ===')
from data.dataloader import get_cifar100_loaders

train_loader, test_loader = get_cifar100_loaders(
    batch_size=4, num_workers=0, data_root='./data', pin_memory=False
)
imgs, labels = next(iter(train_loader))
assert imgs.shape   == (4, 3, 32, 32), f'Train image batch shape: {imgs.shape}'
assert labels.shape == (4,),           f'Train label shape: {labels.shape}'
assert labels.max() < 100,             f'Label out of range: {labels.max()}'
print(f'  Train batch: images {tuple(imgs.shape)}, labels {tuple(labels.shape)}   PASS')

imgs, labels = next(iter(test_loader))
assert imgs.shape == (4, 3, 32, 32)
print(f'  Test  batch: images {tuple(imgs.shape)}, labels {tuple(labels.shape)}   PASS')
print()

# ---------------------------------------------------------------------------
# 4. Forward pass on GPU (if available)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    print('=== GPU forward pass ===')
    device = torch.device('cuda')
    model  = build_model('cbam').to(device)
    dummy  = torch.randn(8, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (8, 100)
    print(f'  CBAM GPU forward pass {tuple(dummy.shape)} -> {tuple(out.shape)}   PASS')
    print()

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print('=' * 50)
print('  All unit tests passed!')
print('=' * 50)
