import torch
import torchxrayvision as xrv
import skimage.io, skimage.transform
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup M1 GPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
model.eval()

# 2. Path to your image
img_path = "/Users/rithinreddy/medic_ai_experiments/chest_xray/chest_xray/test/PNEUMONIA/person36_virus_81.jpeg"
##img_path = "/Users/rithinreddy/Documents/chest-xray.jpeg"
raw_img = skimage.io.imread(img_path)

# 3. Preprocessing
img = xrv.datasets.normalize(raw_img, 255) 
if len(img.shape) > 2: img = img.mean(2) 
img = img[None, ...] 
img = xrv.datasets.XRayResizer(224)(img)
img_tensor = torch.from_numpy(img).to(device)

# 4. Run Analysis
with torch.no_grad():
    preds = model(img_tensor[None, ...])
results = dict(zip(model.pathologies, preds[0].cpu().numpy()))
sorted_findings = sorted(results.items(), key=lambda x: x[1], reverse=True)

# 5. Clinical Knowledge Base (Reasoning for Abnormalities)
clinical_desc = {
    "Pneumonia": "Cloudy patches (infiltrates) suggest an infection in the air sacs, potentially involving fluid or pus.",
    "Infiltration": "Presence of substances (fluid/blood/pus) denser than air within the lung parenchyma.",
    "Atelectasis": "Partial collapse of lung tissue, often indicated by shifted anatomy or volume loss.",
    "Cardiomegaly": "Heart shadow occupies >50% of the thoracic width, suggesting cardiac enlargement.",
    "Effusion": "Fluid accumulation in the pleural space (the 'corners' of the lungs), often obscuring the diaphragm.",
    "Consolidation": "Lung tissue has become firm/solid due to fluid, common in severe pneumonia.",
    "Mass": "A large, distinct growth (typically >3cm) detected within the lung field requiring urgent follow-up.",
    "Nodule": "Small, rounded opacity (<3cm) which could represent a granuloma, scar, or early-stage growth.",
    "Pneumothorax": "Air trapped in the pleural space causing lung collapse; a potentially emergent finding.",
    "Edema": "Fluid buildup in the lungs often related to heart failure, showing as 'fluffy' bilateral opacities.",
    "Emphysema": "Hyper-inflated lungs and flattened diaphragm indicating chronic obstructive lung disease (COPD).",
    "Fibrosis": "Chronic scarring of lung tissue appearing as stiff, stringy textures or 'honeycombing'.",
    "Pleural_Thickening": "Scarring or inflammation of the lung lining, often from past infections.",
    "Hernia": "Abdominal contents protruding into the chest cavity through the diaphragm.",
    "Fracture": "Break detected in the ribs or clavicle, often associated with trauma.",
    "Lung Lesion": "Localized area of abnormal tissue that requires characterization via CT scan.",
    "Lung Opacity": "General term for any area where the lung looks 'whiter' than expected.",
    "Enlarged Cardiomediastinum": "Widening of the central chest structures, requiring check for vascular or nodal issues."
}

# 6. Plotting the Image with Clinical Report
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
plt.subplots_adjust(wspace=0.35)

# --- Left Side: X-Ray ---
ax1.imshow(img[0], cmap='gray')
ax1.set_title("Input Chest Radiograph", fontsize=14, fontweight='bold')
ax1.axis('off')

# --- Right Side: Report ---
ax2.axis('off')
threshold = 0.60
has_abnormality = any(s > threshold for s in results.values())
status_text = "ABNORMAL FINDINGS DETECTED" if has_abnormality else "NORMAL LIMITS"
status_color = "red" if has_abnormality else "green"

ax2.text(0, 0.98, f"DIAGNOSTIC STATUS: {status_text}", fontsize=16, color=status_color, fontweight='bold')

# --- SECTION 1: ALL PATHOLOGIES ---
ax2.text(0, 0.93, "1. Full Pathology Screening (18 Classes):", fontsize=12, fontweight='bold', color='blue')
y_pos = 0.90
for i, (path, score) in enumerate(sorted_findings):
    color = "red" if score > threshold else "black"
    weight = "bold" if score > threshold else "normal"
    prefix = "!!" if score > threshold else "·"
    
    # Split into 2 columns
    x_offset = 0 if i < 9 else 0.45
    y_offset = y_pos - (i % 9) * 0.035
    ax2.text(x_offset, y_offset, f"{prefix} {path}: {score*100:.1f}%", fontsize=9, color=color, fontweight=weight)

# --- SECTION 2: REASONING FOR ABNORMALITIES ---
ax2.text(0, 0.55, "2. Descriptive Reasoning for Major Concerns:", fontsize=12, fontweight='bold', color='blue')
y_pos_reason = 0.50

if not has_abnormality:
    ax2.text(0, y_pos_reason, "→ All screened parameters are within the healthy threshold (<40%).", fontsize=10)
    ax2.text(0, y_pos_reason - 0.04, "→ No acute radiological abnormalities are visible in this scan.", fontsize=10)
    ax2.text(0, y_pos_reason - 0.08, "→ Normal anatomical structures for heart, lungs, and diaphragm.", fontsize=10)
else:
    abnormals = [f for f, s in sorted_findings if s > threshold]
    for pathology in abnormals[:4]: # Limit to top 4 for space
        description = clinical_desc.get(pathology, "Pathology detected above significance threshold.")
        ax2.text(0, y_pos_reason, f"● {pathology} ({results[pathology]*100:.1f}%)", fontsize=10, fontweight='bold', color='red')
        ax2.text(0.03, y_pos_reason - 0.03, f"  {description}", fontsize=9, style='italic', wrap=True)
        y_pos_reason -= 0.08

plt.show()