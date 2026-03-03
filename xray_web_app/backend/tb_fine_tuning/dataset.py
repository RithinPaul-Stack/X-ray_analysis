"""
PyTorch Dataset for 3-Class Chest X-Ray Classification

Handles loading TB, Normal, and Pneumonia chest X-ray images with proper preprocessing.
Supports 3-class classification: Normal (0), TB (1), Pneumonia (2)
"""

import os
import random
from typing import Tuple, List, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

from .config import Config
from .augmentations import get_transforms


class TBXRayDataset(Dataset):
    """
    Dataset for 3-class chest X-ray classification: Normal, TB, Pneumonia.

    Labels:
        0 = Normal
        1 = TB
        2 = Pneumonia

    Args:
        config: Configuration object
        split: One of 'train', 'val', 'test', or 'all'
        transform: Optional transform to apply
    """

    def __init__(
        self,
        config: Config,
        split: str = 'train',
        transform=None
    ):
        self.config = config
        self.split = split
        self.transform = transform

        # Load image paths and labels
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        self._load_dataset()

        # Apply train/val/test split if needed
        if split != 'all':
            self._apply_split()

        print(f"Loaded {len(self)} images for {split} split")
        print(f"  - Normal: {sum(1 for l in self.labels if l == 0)}")
        print(f"  - TB: {sum(1 for l in self.labels if l == 1)}")
        print(f"  - Pneumonia: {sum(1 for l in self.labels if l == 2)}")

    def _load_dataset(self):
        """Load all image paths and labels"""
        # Load TB images (label = 1)
        tb_path = self.config.tb_path
        if os.path.exists(tb_path):
            tb_images = [
                os.path.join(tb_path, f)
                for f in os.listdir(tb_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.image_paths.extend(tb_images)
            self.labels.extend([1] * len(tb_images))
            print(f"Found {len(tb_images)} TB images")

        # Load Normal images (label = 0)
        normal_path = self.config.normal_path
        if os.path.exists(normal_path):
            normal_images = [
                os.path.join(normal_path, f)
                for f in os.listdir(normal_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            # Randomly sample if we have more than max_normal_samples
            if len(normal_images) > self.config.max_normal_samples:
                random.seed(self.config.seed)
                normal_images = random.sample(
                    normal_images,
                    self.config.max_normal_samples
                )
                print(f"Sampled {len(normal_images)} Normal images from available pool")
            else:
                print(f"Found {len(normal_images)} Normal images")

            self.image_paths.extend(normal_images)
            self.labels.extend([0] * len(normal_images))

        # Load Pneumonia images (label = 2)
        pneumonia_path = self.config.pneumonia_path
        if os.path.exists(pneumonia_path):
            pneumonia_images = [
                os.path.join(pneumonia_path, f)
                for f in os.listdir(pneumonia_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            # Randomly sample if we have more than max_pneumonia_samples
            if len(pneumonia_images) > self.config.max_pneumonia_samples:
                random.seed(self.config.seed)
                pneumonia_images = random.sample(
                    pneumonia_images,
                    self.config.max_pneumonia_samples
                )
                print(f"Sampled {len(pneumonia_images)} Pneumonia images from available pool")
            else:
                print(f"Found {len(pneumonia_images)} Pneumonia images")

            self.image_paths.extend(pneumonia_images)
            self.labels.extend([2] * len(pneumonia_images))

    def _apply_split(self):
        """Split dataset into train/val/test"""
        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.image_paths,
            self.labels,
            test_size=self.config.test_ratio,
            random_state=self.config.seed,
            stratify=self.labels
        )

        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_ratio_adjusted,
            random_state=self.config.seed,
            stratify=train_val_labels
        )

        # Select appropriate split
        if self.split == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.split == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        elif self.split == 'test':
            self.image_paths = test_paths
            self.labels = test_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path)

        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            transform = get_transforms(
                self.config.image_size,
                is_training=(self.split == 'train')
            )
            image = transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of each class"""
        return {
            'Normal': sum(1 for l in self.labels if l == 0),
            'TB': sum(1 for l in self.labels if l == 1),
            'Pneumonia': sum(1 for l in self.labels if l == 2)
        }

    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sample weights for weighted random sampling.
        Useful for balancing classes during training.
        """
        class_counts = self.get_class_counts()
        total = len(self.labels)
        num_classes = 3

        # Weight inversely proportional to class frequency
        weights = []
        for label in self.labels:
            if label == 0:  # Normal
                weights.append(total / (num_classes * class_counts['Normal']))
            elif label == 1:  # TB
                weights.append(total / (num_classes * class_counts['TB']))
            else:  # Pneumonia
                weights.append(total / (num_classes * class_counts['Pneumonia']))

        return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(
    config: Config,
    use_weighted_sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Configuration object
        use_weighted_sampling: Use weighted sampling for class balance

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_transform = get_transforms(config.image_size, is_training=True)
    val_transform = get_transforms(config.image_size, is_training=False)

    train_dataset = TBXRayDataset(config, split='train', transform=train_transform)
    val_dataset = TBXRayDataset(config, split='val', transform=val_transform)
    test_dataset = TBXRayDataset(config, split='test', transform=val_transform)

    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    config = Config()
    print(config)

    train_loader, val_loader, test_loader = create_data_loaders(config)

    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
