import torch
import os
import copy
import sys
import csv
import logging
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utility.config import (
    PIN_MEMORY, 
    DATA_DIR, 
    LOG_DIR,
    CSV_DIR,
    BATCH_SIZE, 
    TEST_BATCH_SIZE, 
    IMAGE_SIZE, 
    DATASET_NAME,
    DEVICE
)
from src.layers import QuantizedLayerMixin
from src.evaluation.evaluate import evaluate

logger = logging.getLogger(__name__)

def get_model_size(model):
    """
    Berechnet die Größe des Modells im Arbeitsspeicher in Megabytes (MB).
    Dies dient als theoretischer Vergleichswert.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_data_loaders():
    """
    Die Hauptfunktion zum Laden der Daten.
    """
    if DATASET_NAME == "MNIST":
        return _get_mnist_loaders()
    elif DATASET_NAME == "POKEMON":
        return _get_pokemon_loaders()
    else:
        raise ValueError(f"Unbekanntes Dataset in Config: {DATASET_NAME}")

def _get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download=True ist wichtig für den ersten Run
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    return train_loader, test_loader, 10 # num_classes

def _get_pokemon_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if not os.path.exists(DATA_DIR):
         raise FileNotFoundError(f"Datenverzeichnis nicht gefunden: {DATA_DIR}")
    
    dataset_path = os.path.join(DATA_DIR, "PokemonData") 

    # 1. Den gesamten Datensatz laden (ImageFolder scannt Unterordner als Klassen)
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    total_count = len(full_dataset)
    num_classes = len(full_dataset.classes)
    
    logger.info(f"Gefundene Bilder: {total_count} in {num_classes} Klassen (Pokemon).")

    # 2. Split berechnen (80% Train, 20% Test)
    train_size = int(0.8 * total_count)
    test_size = total_count - train_size
    
    # 3. Random Split durchführen (Seed setzen für Reproduzierbarkeit!)
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    logger.info(f"Split: {train_size} Training, {test_size} Test.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    return train_loader, test_loader, num_classes

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='.')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', color='blue', marker='.')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='green', marker='.')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(LOG_DIR, "Training_Curves.png")
    plt.savefig(save_path)
    plt.close()

def save_csv(results, filename, fieldnames):
    """Hilfsfunktion zum Speichern von Listen in CSV"""
    filepath = os.path.join(CSV_DIR, filename)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info(f"Daten gespeichert unter: {filepath}")

def run_sensitivity_analysis(base_model, test_loader, method='symmetric', bits=8):
    """
    Untersucht Layer-weise Empfindlichkeit:
    Quantisiert immer nur EINEN Layer, lässt alle anderen auf Float.
    """
    logger.info(f"--- Starte Sensitivitätsanalyse (Method: {method}, Bits: {bits}) ---")
    results = []
    
    # Wir brauchen eine saubere Kopie
    model = copy.deepcopy(base_model)
    model.eval()
    model.to(DEVICE)
    
    # 1. Alle Module finden, die wir quantisieren können
    quantizable_modules = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            quantizable_modules.append((name, module))
            
    # 2. Baseline Accuracy messen (sollte der Float-Accuracy entsprechen)
    # Sicherstellen, dass alles auf Float steht
    model.convert_to_baseline()
    base_acc, _ = evaluate(model, test_loader, "Sensitivity Baseline")

    # 3. Schleife durch alle Layer
    for name, module in quantizable_modules:
        # Nur diesen einen Layer quantisieren
        module.prepare_quantization(method=method, bits=bits)
        
        # Evaluieren
        acc, _ = evaluate(model, test_loader, f"Layer: {name}")
        drop = base_acc - acc
        
        results.append({
            "layer_name": name,
            "accuracy": acc,
            "drop": drop
        })
        
        logger.info(f"Layer {name}: Drop = {drop:.2f}%")
        
        # WICHTIG: Layer wieder auf Float zurücksetzen für den nächsten Durchlauf
        module.disable_quantization()
        
    save_csv(results, "sensitivity_analysis.csv", ["layer_name", "accuracy", "drop"])
    return results

def setup_global_logging():
    log_filename = os.path.join(LOG_DIR, "experiment_log.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )