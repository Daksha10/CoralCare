"""
config.py - Central configuration for CoralCare project
Change BASE_DIR to point to your downloaded dataset folder.
"""

# ─── Roboflow ──────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "B9W3xw1u0VdlN5AAqUFh"
ROBOFLOW_WORKSPACE = "coral-pathology-detection-mk1pl"
ROBOFLOW_PROJECT   = "graduation-design-gjwnj"
ROBOFLOW_VERSION   = 1
ROBOFLOW_FORMAT    = "yolov8"

# ─── Dataset paths ─────────────────────────────────────────────────────────────
# After downloading with Roboflow the folder will be created here.
# Update this if you move the dataset elsewhere.
BASE_DIR = "Graduation-Design-1"          # relative to project root

TRAIN_IMAGE_DIR = f"{BASE_DIR}/train/images"
TRAIN_LABEL_DIR = f"{BASE_DIR}/train/labels"

VALID_IMAGE_DIR = f"{BASE_DIR}/valid/images"
VALID_LABEL_DIR = f"{BASE_DIR}/valid/labels"

TEST_IMAGE_DIR  = f"{BASE_DIR}/test/images"
TEST_LABEL_DIR  = f"{BASE_DIR}/test/labels"

# ─── Class mapping ─────────────────────────────────────────────────────────────
CLASS_MAP = {
    0: "bleached",
    1: "unbleached",
    2: "dead"
}

# ─── Model artefacts (saved/loaded by training & app) ──────────────────────────
MODEL_PATH   = "coral_rf_model.pkl"
SCALER_PATH  = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
