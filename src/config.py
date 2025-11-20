import torch
import os

# --- Hardware Konfiguration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
    print(f"Config: Nutze NVIDIA GPU ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False
    print("Config: Nutze CPU")

# --- Pfade ---
# Erstellt automatisch den results Ordner
RESULTS_DIR = "results/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "cnn_float32.pth")
DATA_DIR = "./data"

# --- Hyperparameter ---
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 1.0
EPOCHS = 3
