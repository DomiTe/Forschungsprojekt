import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import matplotlib.pyplot as plt
import os
import copy
from src.utility.logging import get_logger 
from src.utility.quantizer import Quantization
from src.utility.config import PIN_MEMORY, DATA_DIR, BATCH_SIZE, TEST_BATCH_SIZE, IMAGE_SIZE, DATASET_NAME
from src.evaluation.evaluate_model import evaluate
from src.layers import replace_layers_with_quantizable, calibrated_model_activation, QuantizedConv2d, QuantizedLinear

logger = get_logger(__name__)

def get_data_loaders():
    """
    Die Hauptfunktion, die von main.py aufgerufen wird.
    Sie entscheidet, welches Dataset geladen wird.
    """
    if DATASET_NAME == "MNIST":
        return _get_mnist_loaders()
    elif DATASET_NAME == "POKEMON":
        return _get_pokemon_loaders()
    else:
        raise ValueError(f"Unbekanntes Dataset in Config: {DATASET_NAME}")

# --- Interne Hilfsfunktionen (beginnen mit _) ---

def _get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {"num_workers": 0, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    
    return train_loader, test_loader, 10

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

def get_model_size(model_path: str) -> float:
    size = os.path.getsize(model_path)
    return size / (1024 * 1024)

def layer_weight_mse(model_float, model_quant):
    errors = {}
    for (name_f, mod_f), (name_q, mod_q) in zip(model_float.named_modules(), model_quant.named_modules()):
        if isinstance(mod_f, (torch.nn.Conv2d, torch.nn.Linear)) and isinstance(mod_q, (QuantizedConv2d, QuantizedLinear)):
            float_weight = mod_f.weight.data.cpu().float()
            if getattr(mod_q, 'weight_int', None) is None:
                dequant_weight = mod_q.weight.data.cpu().float()
            else:
                dequant_weight = Quantization.dequantize_from_int(mod_q.weight_int.cpu(), mod_q.scale.cpu(), mod_q.zero_point.cpu(), method=mod_q.quant_method)
            mse = torch.mean((float_weight - dequant_weight) ** 2).item()
            errors[name_f] = mse
    return errors

def per_layer_sensitivity_analysis(model_base, test_loader, device, bits=8, method='symmetric'):
    results = []
    
    # Identify layers first
    quantizable_names = [n for n, m in model_base.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]

    logger.info(f"Starting Sensitivity Analysis on {len(quantizable_names)} layers...")

    for layer_name in quantizable_names:
        # 1. Fresh Copy
        model_copy = copy.deepcopy(model_base).to('cpu')
        model_copy = replace_layers_with_quantizable(model_copy)
        
        # 2. Target Specific Layer
        target_found = False
        for name, module in model_copy.named_modules():
            if name == layer_name and hasattr(module, 'quantized_storage'):
                # Quantize Weights for THIS layer
                module.quantized_storage(num_bits=bits, method=method)
                target_found = True
                break
        
        if not target_found:
            continue

        # 3. Calibrate Activations (Required for Full Quantization)
        # We perform a quick calibration on the modified model
        train_loader, _ , _= get_data_loaders()
        calibrated_model_activation(model_copy, train_loader, device='cpu', num_batches=10, method=method, num_bits=bits)

        # 4. Evaluate
        model_copy.to(device)
        acc, inf_time = evaluate(model_copy, test_loader, f"Sensitivity: {layer_name}")
        
        results.append({
            "layer": layer_name,
            "accuracy": acc,
            "inference_time": inf_time
        })
        
        # Cleanup to save RAM
        del model_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 6))

    # Plot 1: Loss (Die Fehlerkurve)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='.')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Accuracy (Das ist der Graph aus deinem Bild!)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', color='blue', marker='.')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange', marker='.')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Plot gespeichert als 'learning_curves.png'")