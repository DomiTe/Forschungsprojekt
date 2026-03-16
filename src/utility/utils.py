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
    elif DATASET_NAME == "CIFAR10":
        return _get_cifar10_loaders()
    elif DATASET_NAME == "CIFAR100":
        return _get_cifar100_loaders()
    elif DATASET_NAME == "FASHION_MNIST":
        return _get_fashion_loaders()
    else:
        raise ValueError(f"Unbekanntes Dataset in Config: {DATASET_NAME}")

def _get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    return train_loader, test_loader, 10 # num_classes

def _get_fashion_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), 
    ])
    
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    return train_loader, test_loader, 10 # num_classes

def _get_cifar10_loaders():
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    
    kwargs = {"num_workers": 2, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    
    logger.info(f"CIFAR-10 loaded: {len(train_dataset)} Train, {len(test_dataset)} Test.")
    
    return train_loader, test_loader, 10 # num_classes

def _get_cifar100_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform_test)
    
    kwargs = {"num_workers": 2, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    
    logger.info(f"CIFAR-100 loaded: {len(train_dataset)} Train, {len(test_dataset)} Test.")
    
    return train_loader, test_loader, 100 # num_classes

def _get_pokemon_loaders():

    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset_path = os.path.join(DATA_DIR, "PokemonData") 
    
    train_dataset_full = datasets.ImageFolder(root=dataset_path, transform=transform_train)
    val_dataset_full = datasets.ImageFolder(root=dataset_path, transform=transform_val)
    
    total_len = len(train_dataset_full)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    
    generator = torch.Generator().manual_seed(42)
    
    train_data, _ = random_split(train_dataset_full, [train_size, val_size], generator=generator)
    
    _, val_data = random_split(val_dataset_full, [train_size, val_size], generator=generator)
    
    kwargs = {"num_workers": 0, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    
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