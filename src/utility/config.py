import torch
import os
from datetime import datetime 

# --- Hardware Konfiguration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False

# --- Timestamp ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Pfade ---
# Erstellt automatisch den results Ordner
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LOG_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# --- Data Paths ---
MODEL_FILENAME = f"cnn_mnist_{TIMESTAMP}.pt"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

QUANTIZED_FILENAME = f"cnn_mnist_quantized{TIMESTAMP}.pt"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, QUANTIZED_FILENAME)

SENSITIVITY_CSV_PATH = os.path.join(CSV_DIR, f"sensitivity_{TIMESTAMP}.csv")
MSE_CSV_PATH = os.path.join(CSV_DIR, f"weight_mse_{TIMESTAMP}.csv")

LOG_FILE_PATH = os.path.join(LOG_DIR, f"experiment_{TIMESTAMP}.log")

DATA_DIR = "./data"

# --- Hyperparameter ---
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 1.0
EPOCHS = 3

KERNEL_SIZE = 3
STRIDE = 1

QUANTIZATION_METHOD = 'symmetric' # Can also be Affine or power of 2(work in progress)
QUANTIZATION_NUM_BITS = 8 # Quantization to 8-bit
QUANTIZATION_NUM_BATCHES = 50 # Batches for quantization calibration