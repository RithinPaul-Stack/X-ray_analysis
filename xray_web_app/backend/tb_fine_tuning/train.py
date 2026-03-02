"""
Training Script for TB Fine-Tuning

Main training loop with validation, early stopping, and model checkpointing.
"""

import os
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from .config import Config
from .model import TBClassifier, create_tb_model, get_optimizer, get_scheduler
from .dataset import create_data_loaders
from .evaluate import evaluate_model, print_metrics


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer class for TB fine-tuning.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create model
        print("\n" + "=" * 50)
        print("Initializing TB Classifier")
        print("=" * 50)
        self.model = create_tb_model(config)

        # Create data loaders
        print("\n" + "=" * 50)
        print("Loading Dataset")
        print("=" * 50)
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config,
            use_weighted_sampling=False  # Using weighted loss instead
        )

        # Loss function with class weights
        class_weights = config.get_class_weights_tensor().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        num_training_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_scheduler(self.optimizer, config, num_training_steps)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )

        # TensorBoard logging
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(config.log_dir, log_name))

        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.current_epoch = 0

        print(f"\nTraining on device: {self.device}")
        print(f"Class weights: {class_weights.tolist()}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update scheduler (if using step-based scheduler)
            if self.config.scheduler == 'cosine':
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__
        }

        # Save latest checkpoint
        latest_path = os.path.join(
            self.config.checkpoint_dir,
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with val_acc: {self.best_val_acc:.2f}%")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed from epoch {self.current_epoch}")

    def train(self) -> Dict[str, float]:
        """
        Full training loop.

        Returns:
            Dictionary with final metrics
        """
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 50 + "\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler (if using epoch-based scheduler)
            if self.config.scheduler in ['step', 'plateau']:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar(
                'LearningRate',
                self.optimizer.param_groups[0]['lr'],
                epoch
            )

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Acc: {self.best_val_acc:.2f}%")

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Load best model and evaluate on test set
        best_path = os.path.join(self.config.checkpoint_dir, 'checkpoint_best.pth')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)

        # Final evaluation
        print("\n" + "=" * 50)
        print("Final Evaluation on Test Set")
        print("=" * 50)
        metrics = evaluate_model(
            self.model,
            self.test_loader,
            self.device,
            self.config.class_names
        )
        print_metrics(metrics)

        # Save final model in simple format
        final_model_path = os.path.join(
            self.config.output_dir,
            'tb_classifier_final.pth'
        )
        self.model.save(final_model_path)

        # Close TensorBoard writer
        self.writer.close()

        return metrics


def main():
    """Main training entry point"""
    # Create config
    config = Config()
    print(config)

    # Create trainer and train
    trainer = Trainer(config)
    metrics = trainer.train()

    print("\n" + "=" * 50)
    print("Training Pipeline Complete!")
    print("=" * 50)
    print(f"\nModel saved to: {config.output_dir}")
    print(f"Logs saved to: {config.log_dir}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir {config.log_dir}")


if __name__ == "__main__":
    main()
