import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import logging

from src.model import CNN
from src.utility.config import (
    DEVICE, 
    EPOCHS, 
    LEARNING_RATE, 
    MODELS_DIR, 
    LOG_DIR
)


logger = logging.getLogger(__name__)

def train_model(train_loader, test_loader, num_classes):
    """
    Trainiert das CNN als Float32-Baseline und speichert es ab.
    """
    logger.info(f"Starte Training auf Device: {DEVICE}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Modell initialisieren
    model = CNN(num_classes=num_classes).to(DEVICE)
    
    # Sicherstellen, dass wir im Float-Modus starten (falls Layers Default anders haben)
    if hasattr(model, 'convert_to_baseline'):
        model.convert_to_baseline()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_train_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoche {epoch+1}/{EPOCHS}")
        
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})

        # Validierung
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct_val / total_val
        
        # Logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        logger.info(f"Ep {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            save_path = os.path.join(MODELS_DIR, "baseline_float32.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model ({train_acc:.4f}%)")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

    save_path = os.path.join(MODELS_DIR, "baseline_float32.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training abgeschlossen. Baseline gespeichert unter: {save_path}")
    
    writer.close()
    
    return model, history