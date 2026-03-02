"""
YOLO-based X-Ray Lesion Detection
Provides bounding box detection for areas of concern in chest X-rays
"""

import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import uuid

# Try to import ultralytics, handle if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO detection will not be available.")


class YOLOAnalyzer:
    """
    YOLO-based chest X-ray lesion detector.
    Identifies and marks areas of concern with bounding boxes.
    """

    def __init__(self):
        """Initialize the YOLO model"""
        self.model = None
        self.model_loaded = False
        self.model_name = "keremberke/yolov8m-chest-xray-classification"

        # Output directory for annotated images
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated')
        os.makedirs(self.output_dir, exist_ok=True)

        # Class labels for chest X-ray findings (common ones)
        self.class_colors = {
            'tb': (0, 0, 255),           # Red for TB
            'tuberculosis': (0, 0, 255),  # Red for TB
            'pneumonia': (0, 165, 255),   # Orange for pneumonia
            'nodule': (0, 255, 255),      # Yellow for nodules
            'mass': (255, 0, 0),          # Blue for mass
            'effusion': (255, 0, 255),    # Magenta for effusion
            'cardiomegaly': (0, 255, 0),  # Green for cardiomegaly
            'default': (0, 0, 255)        # Default red
        }

    def load_model(self):
        """Load the YOLO model (lazy loading)"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics library not installed. Run: pip install ultralytics")

        if not self.model_loaded:
            try:
                print(f"  Loading YOLO model: {self.model_name}")
                # Try to load from Hugging Face Hub
                self.model = YOLO(self.model_name)
                self.model_loaded = True
                print("  YOLO model loaded successfully!")
            except Exception as e:
                # Fallback to a general YOLOv8 model if specific one fails
                print(f"  Warning: Could not load {self.model_name}: {e}")
                print("  Attempting to load default YOLOv8 model...")
                try:
                    self.model = YOLO('yolov8m.pt')
                    self.model_loaded = True
                    print("  Default YOLO model loaded.")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load YOLO model: {e2}")

    def analyze(self, image_path, confidence_threshold=0.25):
        """
        Perform YOLO detection on the X-ray image

        Args:
            image_path: Path to the X-ray image
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary containing detection results and annotated image path
        """
        # Ensure model is loaded
        self.load_model()

        # Read the original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Run YOLO detection
        results = self.model.predict(
            source=image_path,
            conf=confidence_threshold,
            save=False,
            verbose=False
        )

        # Process results
        detections = []
        annotated_image = original_image.copy()

        if len(results) > 0:
            result = results[0]

            # Get boxes if available
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes

                for i, box in enumerate(boxes):
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = result.names.get(class_id, f"Class_{class_id}")

                    # Get color for this class
                    color = self.class_colors.get(class_name.lower(), self.class_colors['default'])

                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                    # Draw label background
                    label = f"{class_name}: {confidence:.1%}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width + 10, y1),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    # Add to detections list
                    detections.append({
                        'class_name': class_name,
                        'confidence': round(confidence * 100, 1),
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    })

        # If no detections with boxes, check for classification results
        if len(detections) == 0 and len(results) > 0:
            result = results[0]

            # Check for classification probabilities
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                top5_indices = probs.top5
                top5_conf = probs.top5conf.cpu().numpy()

                for idx, conf in zip(top5_indices, top5_conf):
                    class_name = result.names.get(idx, f"Class_{idx}")
                    detections.append({
                        'class_name': class_name,
                        'confidence': round(float(conf) * 100, 1),
                        'bbox': None  # Classification only, no bounding box
                    })

                # Add classification results as text overlay
                y_offset = 30
                for det in detections[:3]:  # Top 3
                    label = f"{det['class_name']}: {det['confidence']:.1f}%"
                    cv2.putText(
                        annotated_image,
                        label,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 30

        # Save annotated image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        annotated_filename = f"annotated_{timestamp}_{unique_id}.jpg"
        annotated_path = os.path.join(self.output_dir, annotated_filename)

        cv2.imwrite(annotated_path, annotated_image)

        # Generate summary
        has_findings = len(detections) > 0
        high_confidence_findings = [d for d in detections if d['confidence'] > 50]

        return {
            'success': True,
            'has_findings': has_findings,
            'total_detections': len(detections),
            'high_confidence_count': len(high_confidence_findings),
            'detections': detections,
            'annotated_image': annotated_filename,
            'annotated_path': annotated_path,
            'model_used': self.model_name,
            'summary': self._generate_summary(detections)
        }

    def _generate_summary(self, detections):
        """Generate a human-readable summary of detections"""
        if not detections:
            return "No significant lesions or abnormalities detected by YOLO analysis."

        # Count detections by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = {'count': 0, 'max_confidence': 0}
            class_counts[class_name]['count'] += 1
            class_counts[class_name]['max_confidence'] = max(
                class_counts[class_name]['max_confidence'],
                det['confidence']
            )

        # Build summary
        summary_parts = []
        for class_name, info in sorted(class_counts.items(), key=lambda x: x[1]['max_confidence'], reverse=True):
            if info['count'] == 1:
                summary_parts.append(f"{class_name} ({info['max_confidence']:.1f}% confidence)")
            else:
                summary_parts.append(f"{info['count']} {class_name} regions ({info['max_confidence']:.1f}% max confidence)")

        return f"Detected: {', '.join(summary_parts)}. Areas marked with bounding boxes require clinical correlation."

    def is_available(self):
        """Check if YOLO analysis is available"""
        return YOLO_AVAILABLE


# Singleton instance
_yolo_analyzer = None

def get_yolo_analyzer():
    """Get or create the YOLO analyzer instance"""
    global _yolo_analyzer
    if _yolo_analyzer is None:
        _yolo_analyzer = YOLOAnalyzer()
    return _yolo_analyzer
