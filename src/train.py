import torch
import torch.optim as optim
import torch.nn.functional as F
from src.model import CNN
from src.utility.utils import get_data_loaders
from src.utility.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from src.utility.logging import get_logger
import time

logger = get_logger(__name__)


def train_model() -> CNN:
    logger.info(f"Starting Training on Device: {DEVICE}")
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
                logger.info(
                    f"Train Epoch: {epoch}"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]"
                    f"Loss: {loss.item():.6f}"
                )

        duration = time.time() - start_time
        logger.info(f"Epoch {epoch} finished in {duration:.2f}s")

    model.to("cpu")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Refernce Model saved to: {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train_model()
