import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utility.config import KERNEL_SIZE, STRIDE, CHANNELS, IMAGE_SIZE, NUM_CLASSES


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        dim = IMAGE_SIZE - (KERNEL_SIZE - 1)
        # 2. Nach Conv2
        dim = dim - (KERNEL_SIZE - 1)
        # 3. Nach MaxPool (Faktor 2)
        dim = dim // 2
        # Flatten Size = Pixel_Höhe * Pixel_Breite * Anzahl_Filter (64)
        self.flatten_size = dim * dim * 64
        # Jetzt nutzen wir die Variable statt der festen Zahl 9216
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        # Hier auch NUM_CLASSES statt 10 nutzen!
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
