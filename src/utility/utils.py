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
    if DATASET_NAME == "CIFAR10":
        return _get_cifar10_loaders()
    elif DATASET_NAME == "CIFAR100":
        return _get_cifar100_loaders()
    elif DATASET_NAME == "MNIST":
        return _get_mnist_loaders()
    elif DATASET_NAME == "POKEMON":
        return _get_pokemon_loaders()
    else:
        raise ValueError(f"Unbekanntes Dataset in Config: {DATASET_NAME}")
    
def _get_cifar10_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # COMMENTED OUT
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # COMMENTED OUT
    ])
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    
    kwargs = {"num_workers": 2, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs), \
           DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs), 10

def _get_cifar100_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # COMMENTED OUT
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # COMMENTED OUT
    ])
    train_dataset = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform_test)
    
    kwargs = {"num_workers": 2, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs), \
           DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs), 100
           
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
    # 1. Transform für TRAINING
    # WICHTIG: Wir nutzen jetzt (0.5, 0.5, 0.5) statt der ResNet-Werte
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Nimmt die 64 aus der Config
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standard Normalisierung
    ])

    # 2. Transform für VALIDIERUNG (Keine Augmentation)
    transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Pfad zum Ordner "PokemonData" (wo die 150 Ordner drin sind)
    dataset_path = os.path.join(DATA_DIR, "PokemonData") 
    
    # --- Der Split-Trick (Sauberer Weg) ---
    
    # Wir laden das Dataset ZWEIMAL (einmal mit Train-Transform, einmal mit Val-Transform)
    train_dataset_full = datasets.ImageFolder(root=dataset_path, transform=transform_train)
    val_dataset_full = datasets.ImageFolder(root=dataset_path, transform=transform_val)
    
    # Größe berechnen (80% / 20%)
    total_len = len(train_dataset_full)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    
    # WICHTIG: Denselben Seed nutzen, damit die Indices identisch sind!
    generator = torch.Generator().manual_seed(42)
    
    # Wir splitten BEIDE Datasets identisch
    # train_data nimmt den Teil aus dem Dataset MIT Augmentation
    train_data, _ = random_split(train_dataset_full, [train_size, val_size], generator=generator)
    
    # val_data nimmt den (identischen) Teil aus dem Dataset OHNE Augmentation
    _, val_data = random_split(val_dataset_full, [train_size, val_size], generator=generator)
    
    kwargs = {"num_workers": 0, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    
    # Anzahl Klassen auslesen
    num_classes = len(train_dataset_full.classes)
    
    logger.info(f"Gen-1 Dataset geladen: {len(train_data)} Train, {len(val_data)} Val. Klassen: {num_classes}")
    logger.info(f"Bildgröße: {IMAGE_SIZE}x{IMAGE_SIZE}")

    return train_loader, val_loader, num_classes

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
            
    # 2. Baseline Accuracy messen
    model.convert_to_baseline()
    # KORREKTUR: Dictionary abfangen statt Unpacking
    base_metrics = evaluate(model, test_loader, "Sensitivity Baseline")
    base_acc = base_metrics['accuracy']

    # 3. Schleife durch alle Layer
    for name, module in quantizable_modules:
        # Nur diesen einen Layer quantisieren
        module.prepare_quantization(method=method, bits=bits)
        
        # Evaluieren
        # KORREKTUR: Auch hier das Dictionary abfangen
        eval_metrics = evaluate(model, test_loader, f"Layer: {name}")
        acc = eval_metrics['accuracy']
        
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
    
def save_quantization_plots(results_list):
    """
    Generates and saves separate bar charts for every evaluation metric in the results.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from src.utility.config import LOG_DIR

    df = pd.DataFrame(results_list)
    
    # All metrics to be plotted separately
    metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'sqnr_db', 'kl_divergence', 'avg_mse', 
        'inference_time', 'model_size_mb', 'drop_percentage'
    ]

    plt.style.use('ggplot') # Use a clean, readable style

    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Sort values: Descending for "good" metrics, Ascending for "bad" metrics (errors/time)
        is_error_metric = metric in ['kl_divergence', 'avg_mse', 'inference_time', 'drop_percentage']
        temp_df = df.sort_values(by=metric, ascending=is_error_metric)
        
        bars = plt.bar(temp_df['config_name'], temp_df[metric], color='skyblue', edgecolor='navy')
        
        # Add data labels on top of each bar for precision
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center', fontsize=9)

        plt.title(f"Quantization Analysis: {metric.replace('_', ' ').title()}")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save each plot to results/logs
        plot_path = os.path.join(LOG_DIR, f"plot_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
    
def save_individual_plots(results_list):
    """
    Saves a separate bar chart for every evaluation metric.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(results_list)
    # Define which columns represent plottable metrics
    metrics = ['accuracy', 'sqnr_db', 'kl_divergence', 'avg_mse', 
               'precision', 'recall', 'f1_score', 'inference_time']

    for metric in metrics:
        if metric not in df.columns: continue
            
        plt.figure(figsize=(12, 6))
        # Sort values to make comparisons easier
        temp_df = df.sort_values(by=metric, ascending=False)
        
        plt.bar(temp_df['config_name'], temp_df[metric], color='skyblue', edgecolor='navy')
        plt.title(f"Comparison: {metric.replace('_', ' ').title()}")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(LOG_DIR, f"plot_{metric}.png")
        plt.savefig(plot_path)
        plt.close()