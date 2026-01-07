import torch
import torch.optim as optim
import torch.nn as nn
import logging
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import time

from src.model import CNN
# get_data_loaders hier NICHT mehr importieren, wir bekommen sie übergeben
from src.utility.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, LOG_DIR

logger = logging.getLogger(__name__)

# WICHTIG: Argumente hinzugefügt!
def train_model(train_loader, test_loader, num_classes) -> tuple:
    logger.info(f"Starting Training on Device: {DEVICE}")

    writer = SummaryWriter(log_dir=LOG_DIR)

    model = CNN(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Speicher für die Plot-Daten
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoche {epoch+1}/{EPOCHS}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Accuracy live berechnen
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- VALIDATION PHASE (Neu!) ---
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
        
        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct_val / total_val

        # Werte an TensorBoard & Logger
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        logger.info(f"Ep {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # History speichern für den Plot
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

    # Ende
    model.to("cpu")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Reference Model saved to: {MODEL_SAVE_PATH}")
    
    # Gib Model UND History zurück
    return model, history