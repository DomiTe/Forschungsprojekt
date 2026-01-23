import torch
import os
from datetime import datetime 

# Hardware Configuration 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False

# DATASET Configuration
DATASET_NAME = "POKEMON" # Options: "MNIST" , "CER" , "POKEMON"

if DATASET_NAME == "MNIST":
    IMAGE_SIZE = 28
    CHANNELS = 1       # Black/White
    NUM_CLASSES = 10
    
elif DATASET_NAME == "POKEMON":
    IMAGE_SIZE = 64
    CHANNELS = 3       # RGB
    NUM_CLASSES = 150

# Training Hyperparameter 
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EPOCHS = 200 

# Model Architecture
KERNEL_SIZE = 3
STRIDE = 1

# QUANTIZATION-Configuration 

# List of experiments in a loop
EXPERIMENT_CONFIGS = [
    {"method": "symmetric", "bits": 8, "name": "Sym_INT8"},
    {"method": "symmetric", "bits": 4, "name": "Sym_INT4"},
    {"method": "affine",    "bits": 8, "name": "Aff_INT8"},
    {"method": "affine",    "bits": 4, "name": "Aff_INT4"},
    {"method": "power2",    "bits": 8, "name": "Po2_INT8"},
    {"method": "power2",    "bits": 4, "name": "Po2_INT4"}, # Optional
]

# Configuration for sensitivity layer analysis
SENSITIVITY_CONFIG = {
    "method": "symmetric",
    "bits": 8
}

# Timestamp 
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Paths of Folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

LOG_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
QUANTIZED_MODELS = os.path.join(RESULTS_DIR, "quantized_models")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

# Generating folders if not existing
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(QUANTIZED_MODELS, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Data Paths 
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, f"baseline_float32.pt")

# CSV Data for results
EXPERIMENT_CSV_PATH = os.path.join(CSV_DIR, f"quantization_results_{DATASET_NAME}.csv")
SENSITIVITY_CSV_PATH = os.path.join(CSV_DIR, f"sensitivity_{DATASET_NAME}.csv")
LOG_FILE_PATH = os.path.join(LOG_DIR, f"experiment.log")