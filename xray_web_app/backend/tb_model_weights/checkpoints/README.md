# Model Checkpoints

This directory contains the trained model checkpoint files for the 3-class TB/Pneumonia/Normal classifier.

## Download Checkpoints

The checkpoint files are too large for GitHub (100MB+). You can download them from:

**Google Drive**: [Download Link - To be added]

Or train the model yourself using:

```bash
cd backend/tb_fine_tuning
python train.py
```

## Required Files

After downloading, place these files in this directory:

- `checkpoint_best.pth` (~50MB) - Best model based on validation accuracy (97.30%)
- `checkpoint_latest.pth` (~50MB) - Latest training checkpoint

## Checkpoint Information

**Best Checkpoint (checkpoint_best.pth)**:
- Epoch: 10
- Validation Accuracy: 97.30%
- Validation Loss: 0.1007
- Model: DenseNet-121 (TorchXRayVision)
- Classes: Normal, TB, Pneumonia
- Training Data:
  - Normal: 1,500 samples
  - TB: 700 samples
  - Pneumonia: 1,500 samples

## Verification

After downloading, verify the files:

```bash
# Check file sizes
ls -lh checkpoint_*.pth

# Should show:
# checkpoint_best.pth    ~50MB
# checkpoint_latest.pth  ~50MB
```

## For Windows Users

The checkpoint files will be included in the portable Windows distribution package, so you don't need to download them separately if you're using the pre-packaged version.
