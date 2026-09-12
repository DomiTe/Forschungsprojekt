import torch
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "CIFAR10"

DATASET_SPECS = {
    "CIFAR10":      {"image_size": 32,  "channels": 3, "num_classes": 10},
}

if DATASET_NAME not in DATASET_SPECS:
    raise ValueError(f"Unknown DATASET_NAME '{DATASET_NAME}'. "
                     f"Choose from: {list(DATASET_SPECS)}")

IMAGE_SIZE  = DATASET_SPECS[DATASET_NAME]["image_size"]
CHANNELS    = DATASET_SPECS[DATASET_NAME]["channels"]
NUM_CLASSES = DATASET_SPECS[DATASET_NAME]["num_classes"]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 256
TEST_BATCH_SIZE = 512

LEARNING_RATE   = 0.1        # SGD + cosine schedule works better for ResNets
WEIGHT_DECAY    = 5e-4
MOMENTUM        = 0.9
EPOCHS          = 200

# Use cosine annealing LR schedule (True) or fixed LR (False)
USE_COSINE_LR   = True

PATIENCE = 20
MIN_DELTA = 0.1

# QAT fine-tuning 
QAT_EPOCH = 20
QAT_LR    = 1e-4

# ---------------------------------------------------------------------------
# CNN model architecture 
# ---------------------------------------------------------------------------
KERNEL_SIZE = 3
STRIDE      = 1

# ---------------------------------------------------------------------------
# Hessian hyperparameters
# ---------------------------------------------------------------------------

HESSIAN_BATCH_SIZE = 16


MODEL_NAME = ""

# ---------------------------------------------------------------------------
# Quantization experiments
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = [
    {"method": "power2",    "bits": 8, "name": "Po2_INT8"},
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR         = os.path.join(BASE_DIR, "results")
DATA_DIR            = os.path.join(BASE_DIR, "data")
IMAGENET100_DIR      = os.path.join(DATA_DIR, "imagenet100")  # expects train/ val/ sub-dirs, 100-class subset


RUN_ID = os.environ.get(
    "RUN_ID",
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_local"
)

RUN_DIR             = os.path.join(RESULTS_DIR, RUN_ID)
LOG_DIR             = os.path.join(RUN_DIR, "logs")
MODELS_DIR          = os.path.join(RUN_DIR, "models")
QUANTIZED_MODELS    = os.path.join(RUN_DIR, "quantized_models")
DEPLOYED_MODELS     = os.path.join(RUN_DIR, "deployed_models")
CSV_DIR             = os.path.join(RUN_DIR, "csv")

for _d in (RESULTS_DIR, RUN_DIR, LOG_DIR, MODELS_DIR, QUANTIZED_MODELS, DEPLOYED_MODELS, CSV_DIR):
    os.makedirs(_d, exist_ok=True)

BASELINE_MODEL_PATH     = os.path.join(MODELS_DIR, f"baseline_{MODEL_NAME}_{DATASET_NAME}_float32.pt")
EXPERIMENT_CSV_PATH     = os.path.join(CSV_DIR,    f"results_{MODEL_NAME}_{DATASET_NAME}.csv")
SENSITIVITY_CSV_PATH    = os.path.join(CSV_DIR,    f"sensitivity_{MODEL_NAME}_{DATASET_NAME}.csv")
TIMING_CSV_PATH         = os.path.join(CSV_DIR,    f"timing_{MODEL_NAME}_{DATASET_NAME}.csv")
LOG_FILE_PATH           = os.path.join(LOG_DIR,    "experiment.log")
