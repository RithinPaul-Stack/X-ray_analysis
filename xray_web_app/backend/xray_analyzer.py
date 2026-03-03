"""
X-Ray Analysis Model Wrapper
Encapsulates the torchxrayvision DenseNet model for chest X-ray analysis
Now includes fine-tuned TB classification model
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv
import skimage.io
import numpy as np
import os
import pydicom
from clinical_knowledge import CLINICAL_DESCRIPTIONS, get_severity_level, get_recommendations


# TB Classification Head (same architecture as in training)
class TBClassificationHead(nn.Module):
    """Custom classification head for TB detection."""

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
    """TB Classifier using TorchXRayVision DenseNet-121 as backbone."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = 1024

        # 3-class classification head (Normal, TB, Pneumonia)
        self.tb_head = TBClassificationHead(
            in_features=self.feature_dim,
            hidden_features=512,
            num_classes=3,
            dropout_rate=0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        logits = self.tb_head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class XRayAnalyzer:
    """
    X-Ray Analysis Engine using DenseNet121 trained on multiple chest X-ray datasets.
    Detects 18 different pathologies from chest radiographs.
    """

    def __init__(self):
        """Initialize the model on startup"""
        # Cross-platform device detection: CUDA (Windows/Linux) > MPS (Mac) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("  Using NVIDIA GPU (CUDA)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("  Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("  Using CPU")

        # Load pre-trained DenseNet model
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Analysis threshold for abnormality detection
        self.threshold = 0.5

        # Load fine-tuned TB classifier
        self.tb_classifier = None
        self.tb_available = False
        self._load_tb_model()

    def _load_tb_model(self):
        """Load fine-tuned TB classification model"""
        # Path to fine-tuned weights
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tb_model_weights", "checkpoints", "checkpoint_best.pth"
        )

        if not os.path.exists(weights_path):
            print("  TB model weights not found, TB classification disabled")
            return

        try:
            print("  Loading fine-tuned TB classifier...")

            # Create TB classifier with backbone
            # We need a fresh backbone for TB classifier (not shared with main model)
            tb_backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
            tb_backbone.classifier = nn.Identity()  # Remove original classifier

            self.tb_classifier = TBClassifier(tb_backbone)

            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location='cpu')

            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.tb_classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.tb_classifier.load_state_dict(checkpoint)

            self.tb_classifier = self.tb_classifier.to(self.device)
            self.tb_classifier.eval()
            self.tb_available = True
            print("  TB classifier loaded successfully!")

        except Exception as e:
            print(f"  Failed to load TB classifier: {e}")
            self.tb_classifier = None
            self.tb_available = False

    def analyze_tb(self, img_tensor):
        """
        Perform 3-class classification on preprocessed image (Normal, TB, Pneumonia)

        Args:
            img_tensor: Preprocessed image tensor

        Returns:
            Dictionary with classification results for Normal, TB, and Pneumonia
        """
        if not self.tb_available or self.tb_classifier is None:
            return None

        try:
            with torch.no_grad():
                probs = self.tb_classifier.predict_proba(img_tensor[None, ...])

            normal_prob = float(probs[0][0].cpu().numpy())
            tb_prob = float(probs[0][1].cpu().numpy())
            pneumonia_prob = float(probs[0][2].cpu().numpy())

            # Get prediction (class with highest probability)
            pred_idx = probs[0].argmax().item()
            class_names = ['Normal', 'TB', 'Pneumonia']
            prediction = class_names[pred_idx]
            confidence = float(probs[0][pred_idx].cpu().numpy())

            return {
                'normal_probability': normal_prob,
                'tb_probability': tb_prob,
                'pneumonia_probability': pneumonia_prob,
                'prediction': prediction,
                'confidence': confidence,
                'is_tb_positive': prediction == 'TB',
                'is_pneumonia': prediction == 'Pneumonia'
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return None

    def load_image(self, img_path):
        """
        Load image from various formats (DICOM, PNG, JPEG)

        Args:
            img_path: Path to the image file

        Returns:
            Numpy array of pixel data normalized to 0-255 range
        """
        ext = os.path.splitext(img_path)[1].lower()

        if ext in ['.dcm', '.dicom']:
            # DICOM handling
            dcm = pydicom.dcmread(img_path)
            img = dcm.pixel_array.astype(np.float32)

            # Handle PhotometricInterpretation (invert if needed)
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == 'MONOCHROME1':
                    # MONOCHROME1: white = 0, need to invert
                    img = img.max() - img

            # Apply DICOM windowing if available for better contrast
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                wc = dcm.WindowCenter
                ww = dcm.WindowWidth
                # Handle multi-value window settings
                if isinstance(wc, pydicom.multival.MultiValue):
                    wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue):
                    ww = ww[0]

                # Apply window/level transformation
                img_min = wc - ww / 2
                img_max = wc + ww / 2
                img = np.clip(img, img_min, img_max)
                img = ((img - img_min) / (img_max - img_min)) * 255.0
            else:
                # Normalize based on actual pixel range
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min)) * 255.0
                else:
                    img = np.zeros_like(img)

            return img.astype(np.uint8)
        else:
            # PNG/JPEG handling (existing behavior)
            return skimage.io.imread(img_path)

    def preprocess_image(self, img_path):
        """
        Preprocess X-ray image for model input

        Args:
            img_path: Path to the X-ray image file

        Returns:
            Preprocessed image tensor ready for model inference
        """
        # Read image using unified loader (supports DICOM, PNG, JPEG)
        raw_img = self.load_image(img_path)

        # Normalize to 0-1 range
        img = xrv.datasets.normalize(raw_img, 255)

        # Convert to grayscale if needed (average RGB channels)
        if len(img.shape) > 2:
            img = img.mean(2)

        # Add channel dimension
        img = img[None, ...]

        # Resize to 224x224 (model input size)
        img = xrv.datasets.XRayResizer(224)(img)

        # Convert to tensor and move to device
        img_tensor = torch.from_numpy(img).to(self.device)

        return img_tensor

    def analyze(self, img_path):
        """
        Perform X-ray analysis on the given image

        Args:
            img_path: Path to the X-ray image file

        Returns:
            Dictionary containing analysis results with pathology predictions,
            clinical descriptions, and recommendations
        """
        # Preprocess image
        img_tensor = self.preprocess_image(img_path)

        # Run TB classification first (if available)
        tb_results = self.analyze_tb(img_tensor)

        # Run inference
        with torch.no_grad():
            predictions = self.model(img_tensor[None, ...])

        # Extract results
        pathology_names = self.model.pathologies
        scores = predictions[0].cpu().numpy()

        # Create results dictionary
        results = dict(zip(pathology_names, scores))

        # Sort by score (highest first)
        sorted_findings = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Determine overall status
        has_abnormality = bool(any(float(score) > self.threshold for score in results.values()))
        abnormal_count = int(sum(1 for score in results.values() if float(score) > self.threshold))

        # Build detailed pathology results
        pathologies = []
        for pathology_name, score in sorted_findings:
            score_float = float(score)  # Convert numpy float to Python float
            is_abnormal = bool(score_float > self.threshold)  # Convert to Python bool
            pathologies.append({
                'name': pathology_name,
                'score': score_float,
                'percentage': round(score_float * 100, 1),
                'is_abnormal': is_abnormal,
                'description': CLINICAL_DESCRIPTIONS.get(pathology_name, 'No description available.'),
                'severity': get_severity_level(score_float) if is_abnormal else 'normal'
            })

        # Get top abnormal findings for summary
        abnormal_findings = [p for p in pathologies if p['is_abnormal']]

        # Generate recommendations based on findings
        recommendations = get_recommendations(abnormal_findings)

        result = {
            'status': 'ABNORMAL' if has_abnormality else 'NORMAL',
            'status_message': self._get_status_message(has_abnormality, abnormal_count),
            'threshold': self.threshold,
            'abnormal_count': abnormal_count,
            'total_pathologies': len(pathologies),
            'pathologies': pathologies,
            'top_findings': abnormal_findings[:5],  # Top 5 abnormal findings
            'recommendations': recommendations,
            'disclaimer': self._get_disclaimer()
        }

        # Add TB/Pneumonia classification results if available
        if tb_results:
            result['tb_classification'] = tb_results
            # Update status if TB positive
            if tb_results['is_tb_positive']:
                result['status'] = 'ABNORMAL'
                result['status_message'] = self._get_status_message(True, abnormal_count, tb_positive=True)
                # Add TB-specific recommendation
                if 'TB screening recommended' not in result['recommendations']:
                    result['recommendations'].insert(0,
                        "URGENT: TB screening positive. Immediate referral to pulmonology/infectious disease specialist recommended for confirmatory testing (sputum culture, GeneXpert).")
            # Update status if Pneumonia detected
            elif tb_results.get('is_pneumonia', False):
                result['status'] = 'ABNORMAL'
                result['status_message'] = self._get_status_message(True, abnormal_count, pneumonia_positive=True)
                # Add Pneumonia-specific recommendation
                result['recommendations'].insert(0,
                    "PNEUMONIA DETECTED: Clinical evaluation recommended. Consider chest X-ray follow-up and appropriate antibiotic therapy based on clinical assessment.")

        return result

    def _get_status_message(self, has_abnormality, abnormal_count, tb_positive=False, pneumonia_positive=False):
        """Generate status message based on findings"""
        if tb_positive:
            base_msg = "TB SCREENING POSITIVE - Immediate clinical attention recommended."
            if abnormal_count > 0:
                return f"{base_msg} Additionally, {abnormal_count} other potential abnormalities detected."
            return base_msg
        elif pneumonia_positive:
            base_msg = "PNEUMONIA DETECTED - Clinical evaluation recommended."
            if abnormal_count > 0:
                return f"{base_msg} Additionally, {abnormal_count} other potential abnormalities detected."
            return base_msg
        elif not has_abnormality:
            return "No significant abnormalities detected. All screened parameters are within normal limits."
        elif abnormal_count == 1:
            return "1 potential abnormality detected. Clinical correlation recommended."
        else:
            return f"{abnormal_count} potential abnormalities detected. Clinical correlation strongly recommended."

    def _get_disclaimer(self):
        """Return medical disclaimer"""
        return (
            "DISCLAIMER: This AI-assisted analysis is for informational purposes only and "
            "should not be used as a substitute for professional medical advice, diagnosis, "
            "or treatment. Always consult with a qualified healthcare provider for proper "
            "interpretation of radiological findings."
        )
