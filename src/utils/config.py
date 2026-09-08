"""
Centralized configuration constants for the DR Screening Pipeline.
All thresholds are exposed here so they can be recalibrated without
hunting through module code.
"""

# =============================================================================
# Module 1 — Image Quality Assessment thresholds
# =============================================================================

# Laplacian variance: images below this are considered too blurry to grade.
# Recalibrated for real JPEG fundus cameras (IDRiD, APTOS 2019, Messidor-2).
IQA_SHARPNESS_THRESHOLD = 15.0

# Minimum fraction of the frame that must contain usable retinal tissue
# (measured via green-channel Otsu thresholding). Fundus circle occupies max ~78.5% of square frame.
IQA_FOV_MIN_RATIO = 0.55

# Per-zone exposure bounds (0–255 mean intensity).
IQA_DARK_THRESHOLD = 40.0          # below → underexposed
IQA_BRIGHT_THRESHOLD = 215.0       # above → overexposed
IQA_MACULAR_DARK_FACTOR = 0.7      # macula is naturally darker; scale threshold
IQA_PERIPHERAL_BRIGHT_FACTOR = 1.1 # periphery tolerates slightly more brightness

# Glare: fraction of high-value pixels within the FOV mask.
IQA_GLARE_MAX_RATIO = 0.035

# Borderline multipliers (for ENHANCED_PASS vs hard PASS).
IQA_BORDERLINE_BLUR_FACTOR = 1.5
IQA_BORDERLINE_DARK_FACTOR = 1.3
IQA_BORDERLINE_BRIGHT_FACTOR = 0.85

# CLAHE parameters
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)

# Ben Graham enhancement
BEN_GRAHAM_SIGMA = 30
BEN_GRAHAM_IMG_SIZE = 512

# =============================================================================
# Module 2 — Segmentation thresholds
# =============================================================================

# Optic disc radius as fraction of min(h, w)
SEG_DISC_RADIUS_FRAC = 0.08

# Fovea search offset (fraction of image width)
SEG_FOVEA_OFFSET_FRAC = 0.3
SEG_FOVEA_RADIUS_FRAC = 0.06

# Gabor filter bank parameters for vessel segmentation
GABOR_WAVELENGTH = 8.0
GABOR_SIGMA = 3.0
GABOR_NUM_ORIENTATIONS = 12        # 0° to 165° in 15° steps

# Lesion detection
LESION_MA_MIN_AREA = 3
LESION_MA_MAX_AREA = 45
LESION_MA_MIN_CIRCULARITY = 0.4
LESION_EXUDATE_MIN_AREA = 4
LESION_EXUDATE_MAX_AREA = 600
LESION_HEMORRHAGE_MIN_AREA = 45
LESION_HEMORRHAGE_MAX_AREA = 1200
LESION_EXUDATE_TOPHAT_THRESH = 35

# =============================================================================
# Module 3 — Grading
# =============================================================================

GRADING_MODEL_NAME = "efficientnet_b0"
GRADING_NUM_CLASSES = 5
GRADING_INPUT_SIZE = 224
GRADING_ONNX_OPSET = 17

# ImageNet normalization (standard for timm pretrained models)
GRADING_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
GRADING_NORMALIZE_STD = [0.229, 0.224, 0.225]

# Hybrid fusion weights (NN probability vs clinical rule probability)
GRADING_NN_WEIGHT = 0.7
GRADING_RULE_WEIGHT = 0.3

# ICDR label map
ICDR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

ICDR_DESCRIPTIONS = {
    0: "No retinal lesions detected.",
    1: "Microaneurysms only detected.",
    2: "More than microaneurysms, but less than Severe NPDR (exudates/hemorrhages present).",
    3: "Extensive microaneurysms/hemorrhages in 4 quadrants or venous beading.",
    4: "Neovascularization or vitreous/preretinal hemorrhage present.",
}

# =============================================================================
# Module 4 — Explainability
# =============================================================================

GRADCAM_HEATMAP_ALPHA = 0.4        # overlay weight on original image
GRADCAM_ATTENTION_THRESHOLD = 0.4  # fraction above which pixels count as "attended"

# =============================================================================
# Module 5 — Simulation defaults
# =============================================================================

SIM_DEFAULT_ANNUAL_PATIENTS = 100_000
SIM_DEFAULT_NUM_CAMERAS = 10
SIM_DEFAULT_ACQUISITION_TIME_MIN = 6.0
SIM_DEFAULT_QUALITY_REJECT_RATE = 0.15
SIM_DEFAULT_AI_THROUGHPUT_IMG_MIN = 100.0
SIM_DEFAULT_NUM_DOCTORS = 1
SIM_DEFAULT_DOC_REVIEW_TIME_MIN = 1.5
SIM_DEFAULT_REFERABLE_FRACTION = 0.25  # fraction of AI-processed cases sent to doctor

# =============================================================================
# Phase 4.5 — Offline sync
# =============================================================================

SYNC_RETRY_INTERVAL_SEC = 30
SYNC_MAX_RETRIES = 5
SYNC_BACKOFF_FACTOR = 2.0

# =============================================================================
# Dataset roles (hard constraints — never violated)
# =============================================================================

DATASET_ROLES = {
    "APTOS_2019": "train",       # Training + internal validation (stratified split)
    "IDRiD": "train",            # Combined with APTOS for training
    "Messidor_2": "external",    # External hold-out ONLY — never touched until final eval
    "DRIVE": "vessel_benchmark", # Vessel segmentation scoring ONLY (Dice/Jaccard)
}
