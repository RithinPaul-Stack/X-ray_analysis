"""
Evaluation Module for 3-Class Chest X-Ray Classifier

Provides comprehensive metrics for medical AI evaluation.
Supports 3-class classification: Normal (0), TB (1), Pneumonia (2)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
import matplotlib.pyplot as plt
import os


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str] = ['Normal', 'TB', 'Pneumonia']
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run on
        class_names: Names of classes

    Returns:
        Dictionary with all metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Get predictions
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_preds) * 100

    # Per-class metrics (weighted average across all classes)
    metrics['precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    metrics['recall'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    metrics['f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    # Per-class precision, recall, f1
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0) * 100
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0) * 100
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0) * 100

    # Normal-specific metrics (class 0)
    metrics['normal_precision'] = per_class_precision[0] if len(per_class_precision) > 0 else 0
    metrics['normal_recall'] = per_class_recall[0] if len(per_class_recall) > 0 else 0
    metrics['normal_f1'] = per_class_f1[0] if len(per_class_f1) > 0 else 0

    # TB-specific metrics (class 1)
    metrics['tb_precision'] = per_class_precision[1] if len(per_class_precision) > 1 else 0
    metrics['tb_recall'] = per_class_recall[1] if len(per_class_recall) > 1 else 0
    metrics['tb_f1'] = per_class_f1[1] if len(per_class_f1) > 1 else 0

    # Pneumonia-specific metrics (class 2)
    metrics['pneumonia_precision'] = per_class_precision[2] if len(per_class_precision) > 2 else 0
    metrics['pneumonia_recall'] = per_class_recall[2] if len(per_class_recall) > 2 else 0
    metrics['pneumonia_f1'] = per_class_f1[2] if len(per_class_f1) > 2 else 0

    # Medical terminology
    metrics['sensitivity'] = metrics['tb_recall']  # TB detection rate
    metrics['specificity'] = metrics['normal_recall']  # Normal detection rate

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm

    # ROC-AUC (multi-class One-vs-Rest approach)
    try:
        # Use macro-averaged AUC for multi-class
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
        # Store per-class ROC curves for visualization
        metrics['roc_curve'] = None  # Multi-class ROC requires different handling
    except ValueError as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        metrics['auc_roc'] = 0.0
        metrics['roc_curve'] = None

    # Store raw predictions for further analysis
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print evaluation metrics for 3-class classification"""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS (3-Class: Normal, TB, Pneumonia)")
    print("=" * 60)

    print("\n--- Overall Performance ---")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.2f}% (macro-averaged)")

    print("\n--- Normal Detection (Class 0) ---")
    print(f"  Recall (Specificity): {metrics['normal_recall']:.2f}%")
    print(f"  Precision:            {metrics['normal_precision']:.2f}%")
    print(f"  F1-Score:             {metrics['normal_f1']:.2f}%")

    print("\n--- TB Detection (Class 1) ---")
    print(f"  Recall (Sensitivity): {metrics['tb_recall']:.2f}%")
    print(f"  Precision:            {metrics['tb_precision']:.2f}%")
    print(f"  F1-Score:             {metrics['tb_f1']:.2f}%")

    print("\n--- Pneumonia Detection (Class 2) ---")
    print(f"  Recall:     {metrics['pneumonia_recall']:.2f}%")
    print(f"  Precision:  {metrics['pneumonia_precision']:.2f}%")
    print(f"  F1-Score:   {metrics['pneumonia_f1']:.2f}%")

    print("\n--- Confusion Matrix ---")
    cm = metrics['confusion_matrix']
    print(f"                    Predicted")
    print(f"                Normal    TB    Pneum")
    if cm.shape[0] >= 3 and cm.shape[1] >= 3:
        print(f"  Actual Normal  {cm[0, 0]:5d}  {cm[0, 1]:5d}  {cm[0, 2]:5d}")
        print(f"  Actual TB      {cm[1, 0]:5d}  {cm[1, 1]:5d}  {cm[1, 2]:5d}")
        print(f"  Actual Pneum   {cm[2, 0]:5d}  {cm[2, 1]:5d}  {cm[2, 2]:5d}")
    else:
        print(f"  Matrix shape: {cm.shape}")

    print("\n" + "=" * 60)


def plot_confusion_matrix(
    metrics: Dict[str, float],
    class_names: List[str] = ['Normal', 'TB', 'Pneumonia'],
    save_path: Optional[str] = None
):
    """Plot confusion matrix"""
    cm = metrics['confusion_matrix']

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curve(
    metrics: Dict[str, float],
    save_path: Optional[str] = None
):
    """Plot ROC curve"""
    if metrics['roc_curve'] is None:
        print("ROC curve data not available")
        return

    fpr, tpr, thresholds = metrics['roc_curve']
    auc = metrics['auc_roc'] / 100  # Convert back to 0-1 range

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.close()


def generate_report(
    metrics: Dict[str, float],
    output_dir: str,
    class_names: List[str] = ['Normal', 'TB', 'Pneumonia']
):
    """Generate full evaluation report with plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics, class_names, cm_path)

    # Save ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(metrics, roc_path)

    # Save metrics to text file
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("3-Class Chest X-Ray Classification Report\n")
        f.write("(Normal, TB, Pneumonia)\n")
        f.write("=" * 50 + "\n\n")

        f.write("Overall Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.2f}%\n")
        f.write(f"Precision: {metrics['precision']:.2f}% (weighted)\n")
        f.write(f"Recall:    {metrics['recall']:.2f}% (weighted)\n")
        f.write(f"F1-Score:  {metrics['f1']:.2f}% (weighted)\n")
        f.write(f"AUC-ROC:   {metrics['auc_roc']:.2f}% (macro-averaged)\n\n")

        f.write("Per-Class Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"Normal - Precision: {metrics['normal_precision']:.2f}%, Recall: {metrics['normal_recall']:.2f}%, F1: {metrics['normal_f1']:.2f}%\n")
        f.write(f"TB     - Precision: {metrics['tb_precision']:.2f}%, Recall: {metrics['tb_recall']:.2f}%, F1: {metrics['tb_f1']:.2f}%\n")
        f.write(f"Pneum  - Precision: {metrics['pneumonia_precision']:.2f}%, Recall: {metrics['pneumonia_recall']:.2f}%, F1: {metrics['pneumonia_f1']:.2f}%\n\n")

        f.write("Medical Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(f"TB Sensitivity (Detection Rate): {metrics['sensitivity']:.2f}%\n")
        f.write(f"Specificity (Normal Detection): {metrics['specificity']:.2f}%\n\n")

        f.write("Confusion Matrix\n")
        f.write("-" * 30 + "\n")
        cm = metrics['confusion_matrix']
        f.write("                    Predicted\n")
        f.write("                Normal    TB    Pneum\n")
        if cm.shape[0] >= 3 and cm.shape[1] >= 3:
            f.write(f"  Actual Normal  {cm[0, 0]:5d}  {cm[0, 1]:5d}  {cm[0, 2]:5d}\n")
            f.write(f"  Actual TB      {cm[1, 0]:5d}  {cm[1, 1]:5d}  {cm[1, 2]:5d}\n")
            f.write(f"  Actual Pneum   {cm[2, 0]:5d}  {cm[2, 1]:5d}  {cm[2, 2]:5d}\n")

    print(f"Evaluation report saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from .config import Config
    from .model import TBClassifier
    from .dataset import create_data_loaders

    config = Config()

    # Load model
    model_path = os.path.join(config.output_dir, 'tb_classifier_final.pth')
    if os.path.exists(model_path):
        model = TBClassifier.load(model_path, config)
        model = model.to(config.device)

        # Create test loader
        _, _, test_loader = create_data_loaders(config)

        # Evaluate
        metrics = evaluate_model(
            model,
            test_loader,
            config.device,
            config.class_names
        )

        # Print and save
        print_metrics(metrics)
        generate_report(metrics, config.output_dir, config.class_names)
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
