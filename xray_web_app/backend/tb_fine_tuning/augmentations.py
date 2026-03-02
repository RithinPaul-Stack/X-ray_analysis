"""
Data Augmentation Transforms for TB X-Ray Images

Provides training and validation transforms optimized for chest X-rays.
Note: No vertical flip as it's anatomically incorrect for CXR.
"""

import torch
import torchvision.transforms as T
from typing import Tuple
import random
import numpy as np
from PIL import Image


class ChestXRayAugmentation:
    """
    Augmentation pipeline for chest X-ray images.
    Designed for medical imaging with anatomically correct transforms.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True
    ):
        self.image_size = image_size
        self.is_training = is_training

        if is_training:
            self.transform = T.Compose([
                # Resize to slightly larger for random crop
                T.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),

                # Random crop back to target size
                T.RandomCrop(image_size),

                # Horizontal flip only (vertical is anatomically incorrect)
                T.RandomHorizontalFlip(p=0.5),

                # Rotation (small angles only for X-rays)
                T.RandomRotation(degrees=15),

                # Brightness and contrast adjustments
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),

                # Random affine for slight perspective changes
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),

                # Convert to tensor
                T.ToTensor(),

                # Normalize to [-1024, 1024] range expected by TorchXRayVision
                # Then to [0, 1] for standard training
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # Validation/Test transforms (no augmentation)
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transforms to image"""
        # Ensure grayscale
        if image.mode != 'L':
            image = image.convert('L')

        return self.transform(image)


class TorchXRayVisionTransform:
    """
    Transform specifically designed for TorchXRayVision models.
    Matches the preprocessing expected by the pre-trained models.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True
    ):
        self.image_size = image_size
        self.is_training = is_training

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Process image for TorchXRayVision model.
        Expected input format: grayscale, normalized to [-1024, 1024]
        """
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Convert to numpy
        img_array = np.array(image, dtype=np.float32)

        # Apply augmentations if training
        if self.is_training:
            img_array = self._apply_augmentations(img_array)

        # Resize
        from skimage.transform import resize
        img_array = resize(
            img_array,
            self.image_size,
            mode='constant',
            preserve_range=True
        ).astype(np.float32)

        # Normalize to [-1024, 1024] range (TorchXRayVision standard)
        # Assuming input is 0-255
        img_array = (img_array / 255.0) * 2048 - 1024

        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to numpy array"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            from scipy.ndimage import rotate
            angle = random.uniform(-15, 15)
            img = rotate(img, angle, reshape=False, mode='constant', cval=0)

        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255)

        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255)

        return img


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    use_xrv_format: bool = True
):
    """
    Factory function to get appropriate transforms.

    Args:
        image_size: Target image size
        is_training: Whether to apply augmentations
        use_xrv_format: Use TorchXRayVision format (recommended)

    Returns:
        Transform callable
    """
    if use_xrv_format:
        return TorchXRayVisionTransform(image_size, is_training)
    else:
        return ChestXRayAugmentation(image_size, is_training)
