import torch
import torch.optim as optim
import torch.nn as nn
import logging
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import time

from src.model import CNN
from src.utility.utils import get_data_loaders
from src.utility.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, LOG_DIR


logger = logging.getLogger(__name__)


def train_model() -> CNN:
    logger.info(f"Starting Training on Device: {DEVICE}")

    train_loader, test_loader, num_classes = get_data_loaders()
    writer = SummaryWriter(log_dir=LOG_DIR)

    model = CNN(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        

        pbar = tqdm(train_loader, desc=f"Epoche {epoch+1}/{EPOCHS}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if batch_idx % 100 == 0:
                logger.info(
                    f"Train Epoch: {epoch}"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"Loss: {loss.item():.6f}"
                )

        # Am Ende der Epoche: Durchschnittswerte
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Werte an TensorBoard senden
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        # TODO: Maybe Test-Set evaluieren 

        print(f" -> Epoche {epoch+1} fertig: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    model.to("cpu")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Refernce Model saved to: {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train_model()
