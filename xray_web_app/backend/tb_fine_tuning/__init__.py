"""
TB Fine-Tuning Module for TorchXRayVision DenseNet-121

This module provides tools to fine-tune the pre-trained chest X-ray model
specifically for Tuberculosis (TB) detection.
"""

from .config import Config
from .model import TBClassifier, create_tb_model

__all__ = ['Config', 'TBClassifier', 'create_tb_model']
