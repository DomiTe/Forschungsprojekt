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
# Options: "MNIST" | "FASHION_MNIST" | "CIFAR10" | "CIFAR100" | "POKEMON" | "IMAGENET" | "IMAGENET100"
# ---------------------------------------------------------------------------
DATASET_NAME = "CIFAR10"

DATASET_SPECS = {
    "MNIST":        {"image_size": 28,  "channels": 1, "num_classes": 10},
    "FASHION_MNIST":{"image_size": 28,  "channels": 1, "num_classes": 10},
    "CIFAR10":      {"image_size": 32,  "channels": 3, "num_classes": 10},
    "CIFAR100":     {"image_size": 32,  "channels": 3, "num_classes": 100},
    "POKEMON":      {"image_size": 64,  "channels": 3, "num_classes": 150},
    "IMAGENET":     {"image_size": 224, "channels": 3, "num_classes": 1000},
    "IMAGENET100":  {"image_size": 224, "channels": 3, "num_classes": 100},
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
BATCH_SIZE      = 64
TEST_BATCH_SIZE = 256

# Hessian-vector products (pyhessian) run a double-backward per iteration,
# which is far more memory-hungry than a normal forward/backward pass, so
# the Hessian trace/eigenvalue loaders use a much smaller batch than training.
HESSIAN_BATCH_SIZE = 16

LEARNING_RATE   = 0.1        # SGD + cosine schedule works better for ResNets
WEIGHT_DECAY    = 5e-4
MOMENTUM        = 0.9
EPOCHS          = 3

# Use cosine annealing LR schedule (True) or fixed LR (False)
USE_COSINE_LR   = True

# QAT fine-tuning (src/quantization/train_qat.py) -- matches that module's
# own function defaults (epochs=10, lr=1e-4).
QAT_EPOCH = 10
QAT_LR    = 1e-4

# ---------------------------------------------------------------------------
# CNN model architecture (your existing 4-layer CNN)
# ---------------------------------------------------------------------------
KERNEL_SIZE = 3
STRIDE      = 1

# ---------------------------------------------------------------------------
# Model selection
# Options: "cnn" | "resnet18_scratch" | "resnet18_pretrained" | "resnet50_pretrained" |
#          "resnet18_no_weights" | "resnet50_no_weights"
# ---------------------------------------------------------------------------
MODEL_NAME = "resnet18_scratch"

# ---------------------------------------------------------------------------
# Quantization experiments (unchanged from your original project)
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = [
    {"method": "symmetric", "bits": 8, "name": "Sym_INT8"},
    {"method": "affine",    "bits": 8, "name": "Aff_INT8"},
    {"method": "power2",    "bits": 8, "name": "Po2_INT8"},
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR         = os.path.join(BASE_DIR, "results")
DATA_DIR            = os.path.join(BASE_DIR, "data")
IMAGENET_DIR        = os.path.join(DATA_DIR, "imagenet")      # expects train/ val/ sub-dirs
IMAGENET100_DIR      = os.path.join(DATA_DIR, "imagenet100")  # expects train/ val/ sub-dirs, 100-class subset

# Cluster runs export RUN_ID (timestamp + SLURM_JOB_ID) from the sbatch script
# before invoking torchrun; local runs (plain `python -m src.main ...`, no
# sbatch/torchrun) have no such env var, so fall back to a timestamped local id.
RUN_ID = os.environ.get(
    "RUN_ID",
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_local"
)

# Every artifact from a given invocation -- models, quantized checkpoints,
# deployed int8 models, CSVs, logs -- lives under one results/<RUN_ID>/
# directory, so a later --load-run-id can find everything a given run
# produced in one place (matches results/<RUN_ID>/{csv,logs,...} on disk
# from prior runs).
RUN_DIR             = os.path.join(RESULTS_DIR, RUN_ID)
LOG_DIR             = os.path.join(RUN_DIR, "logs")
MODELS_DIR          = os.path.join(RUN_DIR, "models")
QUANTIZED_MODELS    = os.path.join(RUN_DIR, "quantized_models")
DEPLOYED_MODELS     = os.path.join(RUN_DIR, "deployed_models")
CSV_DIR             = os.path.join(RUN_DIR, "csv")

for _d in (RESULTS_DIR, RUN_DIR, LOG_DIR, MODELS_DIR, QUANTIZED_MODELS, DEPLOYED_MODELS, CSV_DIR):
    os.makedirs(_d, exist_ok=True)

# Per-run saved paths
BASELINE_MODEL_PATH     = os.path.join(MODELS_DIR, f"baseline_{MODEL_NAME}_{DATASET_NAME}_float32.pt")
EXPERIMENT_CSV_PATH     = os.path.join(CSV_DIR,    f"results_{MODEL_NAME}_{DATASET_NAME}.csv")
SENSITIVITY_CSV_PATH    = os.path.join(CSV_DIR,    f"sensitivity_{MODEL_NAME}_{DATASET_NAME}.csv")
TIMING_CSV_PATH         = os.path.join(CSV_DIR,    f"timing_{MODEL_NAME}_{DATASET_NAME}.csv")
LOG_FILE_PATH           = os.path.join(LOG_DIR,    "experiment.log")
