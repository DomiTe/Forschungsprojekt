import torch
import os

# --- Hardware Konfiguration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False

# --- Pfade ---
# Erstellt automatisch den results Ordner
RESULTS_DIR = "results/models"
LOG_DIR = "results/logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "cnn_float32.pth")
DATA_DIR = "./data"

# --- Hyperparameter ---
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 1.0
EPOCHS = 3

KERNEL_SIZE = 3
STRIDE = 1
