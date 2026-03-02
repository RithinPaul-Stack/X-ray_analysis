"""
GradCAM-based X-Ray Region Visualization
Highlights regions that influenced pathology predictions using Gradient-weighted Class Activation Mapping
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
import uuid
import torchxrayvision as xrv
import skimage.io
import pydicom


class GradCAMAnalyzer:
    """
    GradCAM-based chest X-ray region highlighter.
    Uses the TorchXRayVision DenseNet model to visualize which regions
    influenced each pathology prediction.
    """

    def __init__(self, model=None, device=None):
        """
        Initialize the GradCAM analyzer

        Args:
            model: Optional pre-loaded DenseNet model (shares with XRayAnalyzer)
            device: Optional torch device
        """
        # Set device
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Use provided model or load new one
        if model is not None:
            self.model = model
        else:
            print("  Loading DenseNet model for GradCAM...")
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model = self.model.to(self.device)

        self.model.eval()

        # Get pathology names from model
        self.pathologies = self.model.pathologies

        # Output directory for heatmap images
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated')
        os.makedirs(self.output_dir, exist_ok=True)

        # Threshold for abnormality detection
        self.threshold = 0.5

        # Storage for hooks
        self.gradients = None
        self.activations = None

        # Color map for different pathologies (using distinct colors)
        self.pathology_colors = {
            'Atelectasis': (255, 0, 0),      # Red
            'Cardiomegaly': (0, 255, 0),     # Green
            'Effusion': (0, 0, 255),         # Blue
            'Infiltration': (255, 255, 0),   # Yellow
            'Mass': (255, 0, 255),           # Magenta
            'Nodule': (0, 255, 255),         # Cyan
            'Pneumonia': (255, 128, 0),      # Orange
            'Pneumothorax': (128, 0, 255),   # Purple
            'Consolidation': (255, 64, 64),  # Light Red
            'Edema': (64, 255, 64),          # Light Green
            'Emphysema': (64, 64, 255),      # Light Blue
            'Fibrosis': (255, 128, 128),     # Pink
            'Pleural_Thickening': (128, 255, 128),  # Light Green
            'Hernia': (128, 128, 255),       # Light Purple
        }

    def _register_hooks(self, target_layer):
        """Register forward and backward hooks on the target layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        return forward_handle, backward_handle

    def _get_target_layer(self):
        """Get the last convolutional layer for GradCAM"""
        # DenseNet121 structure: features -> denseblock4 -> norm5
        # We target the last dense block's output
        return self.model.features.denseblock4

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
                    img = img.max() - img

            # Apply DICOM windowing if available
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                wc = dcm.WindowCenter
                ww = dcm.WindowWidth
                if isinstance(wc, pydicom.multival.MultiValue):
                    wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue):
                    ww = ww[0]

                img_min = wc - ww / 2
                img_max = wc + ww / 2
                img = np.clip(img, img_min, img_max)
                img = ((img - img_min) / (img_max - img_min)) * 255.0
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min)) * 255.0
                else:
                    img = np.zeros_like(img)

            return img.astype(np.uint8)
        else:
            return skimage.io.imread(img_path)

    def preprocess_image(self, img_path):
        """Preprocess X-ray image for model input"""
        # Read image using unified loader
        raw_img = self.load_image(img_path)

        # Store original for overlay
        self.original_image = raw_img.copy()

        # Normalize to 0-1 range
        img = xrv.datasets.normalize(raw_img, 255)

        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = img.mean(2)

        # Add channel dimension
        img = img[None, ...]

        # Resize to 224x224
        img = xrv.datasets.XRayResizer(224)(img)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().to(self.device)

        return img_tensor

    def compute_gradcam(self, img_tensor, target_class_idx):
        """
        Compute GradCAM heatmap for a specific class

        Args:
            img_tensor: Preprocessed image tensor
            target_class_idx: Index of the target pathology class

        Returns:
            Normalized heatmap as numpy array
        """
        # Get target layer
        target_layer = self._get_target_layer()

        # Register hooks
        forward_handle, backward_handle = self._register_hooks(target_layer)

        try:
            # Forward pass
            self.model.zero_grad()
            output = self.model(img_tensor[None, ...])

            # Backward pass for target class
            target = output[0, target_class_idx]
            target.backward(retain_graph=True)

            # Get gradients and activations
            gradients = self.gradients
            activations = self.activations

            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

            # Weighted combination of activations
            cam = torch.sum(weights * activations, dim=1, keepdim=True)

            # ReLU to keep only positive contributions
            cam = F.relu(cam)

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam

        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()

    def create_heatmap_overlay(self, original_image, heatmap, alpha=0.4):
        """
        Create a heatmap overlay on the original image

        Args:
            original_image: Original X-ray image
            heatmap: GradCAM heatmap (normalized 0-1)
            alpha: Transparency of the heatmap overlay

        Returns:
            Blended image with heatmap overlay
        """
        # Ensure original is RGB
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        else:
            original_rgb = original_image.copy()

        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))

        # Apply colormap (TURBO gives good medical visualization)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_TURBO
        )

        # Convert to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Ensure original is uint8
        if original_rgb.dtype != np.uint8:
            original_rgb = (original_rgb * 255).astype(np.uint8) if original_rgb.max() <= 1 else original_rgb.astype(np.uint8)

        # Blend images
        blended = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)

        return blended

    def analyze(self, image_path, analysis_results=None):
        """
        Perform GradCAM analysis on the X-ray image

        Args:
            image_path: Path to the X-ray image
            analysis_results: Optional pre-computed analysis results from XRayAnalyzer

        Returns:
            Dictionary containing heatmap visualization results
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)

        # Run inference to get predictions
        with torch.no_grad():
            predictions = self.model(img_tensor[None, ...])

        scores = predictions[0].cpu().numpy()

        # Find pathologies above threshold
        detections = []
        combined_heatmap = None

        for idx, (pathology, score) in enumerate(zip(self.pathologies, scores)):
            score_float = float(score)

            if score_float > self.threshold:
                # Compute GradCAM for this pathology
                heatmap = self.compute_gradcam(img_tensor, idx)

                # Accumulate heatmaps (weighted by confidence)
                if combined_heatmap is None:
                    combined_heatmap = heatmap * score_float
                else:
                    combined_heatmap = np.maximum(combined_heatmap, heatmap * score_float)

                detections.append({
                    'class_name': pathology,
                    'confidence': round(score_float * 100, 1),
                    'bbox': None  # GradCAM doesn't produce bounding boxes
                })

        # If no abnormalities detected, create a neutral visualization
        if combined_heatmap is None:
            # Show top prediction's heatmap even if below threshold
            top_idx = np.argmax(scores)
            combined_heatmap = self.compute_gradcam(img_tensor, top_idx)
            detections.append({
                'class_name': self.pathologies[top_idx],
                'confidence': round(float(scores[top_idx]) * 100, 1),
                'bbox': None
            })

        # Normalize combined heatmap
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)

        # Create overlay image
        annotated_image = self.create_heatmap_overlay(self.original_image, combined_heatmap)

        # Save annotated image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        annotated_filename = f"heatmap_{timestamp}_{unique_id}.jpg"
        annotated_path = os.path.join(self.output_dir, annotated_filename)

        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Sort detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Generate summary
        has_findings = any(d['confidence'] > self.threshold * 100 for d in detections)
        high_confidence = [d for d in detections if d['confidence'] > 50]

        return {
            'success': True,
            'has_findings': has_findings,
            'total_detections': len(detections),
            'high_confidence_count': len(high_confidence),
            'detections': detections,
            'annotated_image': annotated_filename,
            'annotated_path': annotated_path,
            'model_used': 'DenseNet121-GradCAM',
            'summary': self._generate_summary(detections)
        }

    def _generate_summary(self, detections):
        """Generate a human-readable summary of findings"""
        if not detections:
            return "No significant regions highlighted. The X-ray appears normal."

        # Filter to significant findings
        significant = [d for d in detections if d['confidence'] > self.threshold * 100]

        if not significant:
            top = detections[0]
            return f"Highest attention region: {top['class_name']} ({top['confidence']:.1f}% confidence). No abnormalities above threshold detected."

        # Build summary
        findings = [f"{d['class_name']} ({d['confidence']:.1f}%)" for d in significant[:5]]

        return f"Highlighted regions for: {', '.join(findings)}. Heatmap shows areas that influenced the AI's predictions. Clinical correlation recommended."

    def is_available(self):
        """Check if GradCAM analysis is available"""
        return True


# Singleton instance
_gradcam_analyzer = None

def get_gradcam_analyzer(model=None, device=None):
    """Get or create the GradCAM analyzer instance"""
    global _gradcam_analyzer
    if _gradcam_analyzer is None:
        _gradcam_analyzer = GradCAMAnalyzer(model=model, device=device)
    return _gradcam_analyzer

# For backwards compatibility with existing imports
GRADCAM_AVAILABLE = True
