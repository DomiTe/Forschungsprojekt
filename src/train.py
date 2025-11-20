import torch
import torch.optim as optim
import torch.nn.functional as F
from src.model import CNN
from src.utils import get_data_loaders
from src.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
import time


def train_model() -> CNN:
    train_loader, test_loader = get_data_loaders()
    model = CNN().to(DEVICE)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch}[{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}"
                )
        print(f"Epoch {epoch} beendet in {time.time() - start_time:.2f}s")
    model.to("cpu")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Referenzmodell gespeichert unter: {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train_model()
