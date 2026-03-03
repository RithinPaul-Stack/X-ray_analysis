"""
Configuration for TB Fine-Tuning

Contains all hyperparameters, paths, and settings for training.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List
import torch


@dataclass
class Config:
    """Configuration class for TB fine-tuning"""

    # Dataset paths - TB and Normal from TB dataset
    # Uses environment variables or defaults to user's Documents folder (cross-platform)
    data_root: str = field(default_factory=lambda: os.environ.get(
        "TB_DATA_ROOT",
        os.path.join(os.path.expanduser("~"), "Documents", "TB_Chest_Radiography_Database")
    ))
    tb_folder: str = "Tuberculosis"
    normal_folder: str = "Normal"

    # Pneumonia dataset path
    pneumonia_data_root: str = field(default_factory=lambda: os.environ.get(
        "PNEUMONIA_DATA_ROOT",
        os.path.join(os.path.expanduser("~"), "Documents", "chest_xray", "train")
    ))
    pneumonia_folder: str = "PNEUMONIA"

    # Output paths
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tb_model_weights"
    )
    checkpoint_dir: str = field(default="")
    log_dir: str = field(default="")

    # Dataset settings - limit samples for class balance
    max_normal_samples: int = 1500  # Limit normal images for better class balance
    max_pneumonia_samples: int = 1500  # Limit pneumonia images for balance with TB (700)
    image_size: Tuple[int, int] = (224, 224)  # TorchXRayVision expects 224x224

    # Data split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer settings
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 3

    # Early stopping
    patience: int = 7
    min_delta: float = 0.001

    # Class weights for imbalanced data (Normal=0, TB=1, Pneumonia=2)
    # With 1500 Normal, 700 TB, 1500 Pneumonia - TB is minority class
    # Weight = total_samples / (num_classes * class_samples)
    # TB weight = 3700 / (3 * 700) = 1.76, normalized to ~2.14 relative to others
    class_weights: Tuple[float, float, float] = (1.0, 2.14, 1.0)

    # Model settings
    freeze_backbone: bool = True  # Freeze early layers
    unfreeze_last_n_blocks: int = 1  # Unfreeze last N dense blocks
    dropout_rate: float = 0.5

    # Device settings
    device: str = field(default="")
    num_workers: int = 4
    pin_memory: bool = True

    # Random seed for reproducibility
    seed: int = 42

    # Class names (3-class classification)
    class_names: List[str] = field(default_factory=lambda: ["Normal", "TB", "Pneumonia"])
    num_classes: int = 3

    def __post_init__(self):
        """Initialize derived paths and device"""
        # Set device - cross-platform: CUDA (Windows/Linux) > MPS (Mac) > CPU
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Set derived paths
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def tb_path(self) -> str:
        """Full path to TB images"""
        return os.path.join(self.data_root, self.tb_folder)

    @property
    def normal_path(self) -> str:
        """Full path to Normal images"""
        return os.path.join(self.data_root, self.normal_folder)

    @property
    def pneumonia_path(self) -> str:
        """Full path to Pneumonia images"""
        return os.path.join(self.pneumonia_data_root, self.pneumonia_folder)

    def get_class_weights_tensor(self) -> torch.Tensor:
        """Get class weights as a tensor on the correct device"""
        return torch.tensor(self.class_weights, dtype=torch.float32)

    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = ["=" * 50, "TB Fine-Tuning Configuration", "=" * 50]
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                lines.append(f"  {key}: {value}")
        lines.append("=" * 50)
        return "\n".join(lines)


# Default configuration instance
default_config = Config()
