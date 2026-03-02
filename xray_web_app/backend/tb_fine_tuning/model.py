"""
TB Classifier Model based on TorchXRayVision DenseNet-121

Fine-tunes the pre-trained chest X-ray model for TB classification.
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv
from typing import Optional, Dict, Any
import os

from .config import Config


class TBClassificationHead(nn.Module):
    """
    Custom classification head for TB detection.
    Replaces the original 18-class output with binary classification.
    """

    def __init__(
        self,
        in_features: int = 1024,
        hidden_features: int = 512,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TBClassifier(nn.Module):
    """
    TB Classifier using TorchXRayVision DenseNet-121 as backbone.

    Freezes early layers and adds a custom classification head.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # Load pre-trained TorchXRayVision model
        print("Loading pre-trained TorchXRayVision DenseNet-121...")
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")

        # Get the feature dimension from backbone
        # DenseNet-121 has 1024 features before the classifier
        self.feature_dim = 1024

        # Remove the original classifier
        # The backbone outputs features, we'll add our own head
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if configured
        if config.freeze_backbone:
            self._freeze_backbone()

        # Add TB classification head
        self.tb_head = TBClassificationHead(
            in_features=self.feature_dim,
            hidden_features=512,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        )

        print(f"Model initialized with {self._count_parameters()} trainable parameters")

    def _freeze_backbone(self):
        """Freeze backbone layers except last N dense blocks"""
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N dense blocks if specified
        if self.config.unfreeze_last_n_blocks > 0:
            # DenseNet-121 has: features.denseblock1, 2, 3, 4
            dense_blocks = ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1']
            blocks_to_unfreeze = dense_blocks[:self.config.unfreeze_last_n_blocks]

            for name, param in self.backbone.named_parameters():
                for block_name in blocks_to_unfreeze:
                    if block_name in name:
                        param.requires_grad = True
                        break

        # Always unfreeze batch norm layers for better adaptation
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 224, 224)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Get features from backbone
        features = self.backbone.features(x)

        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        # Classification
        logits = self.tb_head(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities of shape (batch, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor

        Returns:
            Predicted labels of shape (batch,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representations from backbone.

        Args:
            x: Input tensor

        Returns:
            Features of shape (batch, feature_dim)
        """
        features = self.backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features

    def save(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, config: Optional[Config] = None) -> 'TBClassifier':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')

        if config is None:
            config = Config(**checkpoint['config'])

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

        return model


def create_tb_model(config: Optional[Config] = None) -> TBClassifier:
    """
    Factory function to create TB classifier model.

    Args:
        config: Configuration object (uses default if None)

    Returns:
        TBClassifier model
    """
    if config is None:
        config = Config()

    model = TBClassifier(config)
    model = model.to(config.device)

    return model


def get_optimizer(model: TBClassifier, config: Config) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for backbone and head.
    """
    # Separate parameters
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'tb_head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    # Different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': config.learning_rate}  # Normal LR for head
    ]

    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.weight_decay
        )

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
    num_training_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.num_epochs // 3,
            gamma=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

    return scheduler


if __name__ == "__main__":
    # Test model creation
    config = Config()
    print(config)

    model = create_tb_model(config)
    print(f"\nModel on device: {config.device}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 224, 224).to(config.device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    probs = model.predict_proba(dummy_input)
    print(f"Probabilities: {probs}")
