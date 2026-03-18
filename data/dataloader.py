"""
CIFAR-100 data loading with standard augmentation.

Train transforms:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  - ToTensor + Normalize

Test transforms:
  - ToTensor + Normalize

Dataset statistics (channel-wise mean / std computed over training set):
  mean = (0.5071, 0.4867, 0.4408)
  std  = (0.2675, 0.2565, 0.2761)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = './data',
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return CIFAR-100 train and test DataLoaders.

    Args:
        batch_size:  Mini-batch size.
        num_workers: Parallel data-loading workers.
        data_root:   Directory where the dataset will be downloaded / cached.
        pin_memory:  Pin memory for faster GPU transfer (set False for CPU-only).

    Returns:
        (train_loader, test_loader)
    """
    normalize = transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR100(
        root=data_root, train=True,  download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader
