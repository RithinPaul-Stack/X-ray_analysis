"""
Clinical Knowledge Base
Contains medical descriptions, severity classifications, and recommendations
for X-ray pathology findings
"""

# Clinical descriptions for each pathology
CLINICAL_DESCRIPTIONS = {
    "Atelectasis": (
        "Partial or complete collapse of lung tissue. This appears as increased opacity "
        "with volume loss, potentially showing shifted fissures, elevated hemidiaphragm, "
        "or mediastinal shift toward the affected side. Common causes include mucus plugging, "
        "post-surgical changes, or external compression."
    ),
    "Cardiomegaly": (
        "Enlargement of the cardiac silhouette, typically defined as a cardiothoracic ratio "
        "exceeding 50% on a PA chest radiograph. May indicate underlying conditions such as "
        "heart failure, valvular disease, cardiomyopathy, or pericardial effusion."
    ),
    "Consolidation": (
        "Replacement of air in the alveoli with fluid, pus, blood, or cells, appearing as "
        "dense opacity that may contain air bronchograms. Commonly seen in pneumonia, "
        "pulmonary hemorrhage, or acute respiratory distress syndrome (ARDS)."
    ),
    "Edema": (
        "Accumulation of fluid in the lung interstitium and alveoli. Characterized by "
        "bilateral perihilar haziness, Kerley B lines, peribronchial cuffing, and pleural "
        "effusions. Often associated with heart failure or fluid overload states."
    ),
    "Effusion": (
        "Accumulation of fluid in the pleural space between the lung and chest wall. "
        "Appears as blunting of the costophrenic angle on upright films or as a dependent "
        "opacity on supine radiographs. May be transudative or exudative in nature."
    ),
    "Emphysema": (
        "Chronic lung condition characterized by destruction of alveolar walls resulting in "
        "hyperinflation. Radiographic signs include flattened diaphragms, increased "
        "retrosternal airspace, and hyperlucent lung fields with diminished vascular markings."
    ),
    "Fibrosis": (
        "Scarring and thickening of lung tissue appearing as reticular or reticulonodular "
        "opacities, often with volume loss. Advanced cases may show honeycombing pattern. "
        "Can result from chronic inflammation, radiation, or idiopathic pulmonary fibrosis."
    ),
    "Hernia": (
        "Protrusion of abdominal contents into the thoracic cavity through a defect in the "
        "diaphragm. May appear as a retrocardiac opacity with air-fluid levels. Types include "
        "hiatal hernia, Bochdalek hernia, and traumatic diaphragmatic hernia."
    ),
    "Infiltration": (
        "Presence of abnormal substances (inflammatory cells, fluid, or other material) "
        "within the lung parenchyma appearing as hazy opacification. Often used as a general "
        "term for non-specific pulmonary opacities requiring clinical correlation."
    ),
    "Mass": (
        "A discrete pulmonary opacity greater than 3 cm in diameter. Requires urgent "
        "evaluation to characterize and determine if malignant. Features such as spiculated "
        "margins, cavitation, or associated lymphadenopathy may suggest malignancy."
    ),
    "Nodule": (
        "A rounded opacity less than 3 cm in diameter. May be benite (granuloma, hamartoma) "
        "or malignant. Size, growth rate, margins, and calcification patterns help determine "
        "likelihood of malignancy. Follow-up imaging often recommended."
    ),
    "Pleural_Thickening": (
        "Thickening of the pleural membrane, appearing as a smooth or irregular opacity "
        "along the chest wall or fissures. May result from prior infection, inflammation, "
        "asbestos exposure, or malignancy (mesothelioma)."
    ),
    "Pneumonia": (
        "Infection of the lung parenchyma presenting as focal or diffuse opacification. "
        "May show air bronchograms, lobar distribution, or patchy infiltrates depending on "
        "the causative organism. Clinical symptoms include fever, cough, and dyspnea."
    ),
    "Pneumothorax": (
        "Presence of air in the pleural space causing partial or complete lung collapse. "
        "Appears as a visible pleural line with absent lung markings peripherally. "
        "Tension pneumothorax is a medical emergency requiring immediate intervention."
    ),
    "Enlarged Cardiomediastinum": (
        "Widening of the mediastinal silhouette beyond normal limits. May indicate aortic "
        "aneurysm, mediastinal mass, lymphadenopathy, or mediastinal hemorrhage. "
        "Comparison with prior imaging and clinical correlation essential."
    ),
    "Lung Opacity": (
        "General term for any area of increased density in the lung fields. Requires "
        "characterization as alveolar, interstitial, or mixed pattern. Clinical context "
        "and comparison with prior studies help narrow the differential diagnosis."
    ),
    "Lung Lesion": (
        "Localized abnormality within the lung parenchyma. Further characterization with "
        "CT imaging is often necessary to determine nature (solid, ground-glass, cavitary) "
        "and guide appropriate management or biopsy."
    ),
    "Fracture": (
        "Break in bone continuity, most commonly involving ribs or clavicle on chest "
        "radiograph. May be associated with trauma, pathologic processes, or stress. "
        "Rib fractures require evaluation for potential pneumothorax or hemothorax."
    ),
    "Support Devices": (
        "Medical devices visible on radiograph including endotracheal tubes, central lines, "
        "pacemakers, chest tubes, and feeding tubes. Position verification is important to "
        "ensure proper placement and function."
    ),
    "No Finding": (
        "No significant radiographic abnormality detected on the chest radiograph. "
        "Lungs are clear, cardiac silhouette is normal, and no acute osseous abnormality "
        "is identified. Clinical correlation always recommended."
    )
}


def get_severity_level(score):
    """
    Determine severity level based on prediction score

    Args:
        score: Model prediction score (0.0 to 1.0)

    Returns:
        String indicating severity level
    """
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "moderate"
    elif score >= 0.5:
        return "low"
    else:
        return "normal"


def get_severity_color(severity):
    """
    Get color code for severity level

    Args:
        severity: Severity level string

    Returns:
        Color code string
    """
    colors = {
        "high": "#dc2626",      # Red
        "moderate": "#f97316",  # Orange
        "low": "#eab308",       # Yellow
        "normal": "#22c55e"     # Green
    }
    return colors.get(severity, "#6b7280")


def get_recommendations(abnormal_findings):
    """
    Generate clinical recommendations based on findings

    Args:
        abnormal_findings: List of abnormal pathology findings

    Returns:
        List of recommendation strings
    """
    if not abnormal_findings:
        return [
            "No significant abnormalities detected.",
            "Routine follow-up as clinically indicated.",
            "Maintain preventive care and regular health screenings."
        ]

    recommendations = []
    high_severity = [f for f in abnormal_findings if f.get('severity') == 'high']
    moderate_severity = [f for f in abnormal_findings if f.get('severity') == 'moderate']

    # General recommendation
    recommendations.append(
        "Clinical correlation with patient history and physical examination is recommended."
    )

    # Severity-based recommendations
    if high_severity:
        recommendations.append(
            f"HIGH PRIORITY: {len(high_severity)} finding(s) with high confidence scores "
            "require urgent clinical attention and possible additional imaging."
        )

    if moderate_severity:
        recommendations.append(
            f"MODERATE PRIORITY: {len(moderate_severity)} finding(s) should be correlated "
            "with clinical presentation and may warrant follow-up imaging."
        )

    # Specific condition recommendations
    pathology_names = [f['name'] for f in abnormal_findings]

    if 'Pneumothorax' in pathology_names:
        recommendations.append(
            "URGENT: If pneumothorax is confirmed, immediate clinical assessment for "
            "respiratory distress is required. Consider chest tube placement if significant."
        )

    if 'Mass' in pathology_names:
        recommendations.append(
            "IMPORTANT: Pulmonary mass detected. CT chest with contrast recommended for "
            "further characterization. Consider pulmonology/oncology referral."
        )

    if 'Pneumonia' in pathology_names or 'Consolidation' in pathology_names:
        recommendations.append(
            "Consider antibiotic therapy if bacterial pneumonia is suspected clinically. "
            "Follow-up imaging in 4-6 weeks to confirm resolution."
        )

    if 'Cardiomegaly' in pathology_names or 'Edema' in pathology_names:
        recommendations.append(
            "Cardiology consultation recommended. Consider echocardiogram and BNP levels "
            "to evaluate cardiac function."
        )

    if 'Effusion' in pathology_names:
        recommendations.append(
            "If effusion is large or symptomatic, consider thoracentesis for diagnostic "
            "and therapeutic purposes."
        )

    # Standard footer
    recommendations.append(
        "This AI analysis should be reviewed by a qualified radiologist and treating physician."
    )

    return recommendations


# Pathology categories for UI grouping
PATHOLOGY_CATEGORIES = {
    "Infectious/Inflammatory": ["Pneumonia", "Consolidation", "Infiltration"],
    "Cardiac": ["Cardiomegaly", "Enlarged Cardiomediastinum", "Edema"],
    "Pleural": ["Effusion", "Pleural_Thickening", "Pneumothorax"],
    "Parenchymal": ["Atelectasis", "Emphysema", "Fibrosis"],
    "Masses/Nodules": ["Mass", "Nodule", "Lung Lesion", "Lung Opacity"],
    "Other": ["Hernia", "Fracture", "Support Devices", "No Finding"]
}
