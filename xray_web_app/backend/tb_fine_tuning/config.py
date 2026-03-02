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

    # Dataset paths
    data_root: str = "/Users/rithinreddy/Documents/TB_Chest_Radiography_Database"
    tb_folder: str = "Tuberculosis"
    normal_folder: str = "Normal"

    # Output paths
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tb_model_weights"
    )
    checkpoint_dir: str = field(default="")
    log_dir: str = field(default="")

    # Dataset settings
    max_normal_samples: int = 2000  # Limit normal images for better class balance
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

    # Class weights for imbalanced data (Normal=0, TB=1)
    # With 2000 Normal and 700 TB, ratio is ~2.86:1
    class_weights: Tuple[float, float] = (1.0, 2.86)

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

    # Class names
    class_names: List[str] = field(default_factory=lambda: ["Normal", "TB"])
    num_classes: int = 2

    def __post_init__(self):
        """Initialize derived paths and device"""
        # Set device
        if not self.device:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
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
