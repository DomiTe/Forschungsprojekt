import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utility.config import KERNEL_SIZE, STRIDE, CHANNELS, IMAGE_SIZE, NUM_CLASSES
from src.layers import QuantizedConv2d, QuantizedLinear

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        # Ersetze nn.Conv2d durch QuantizedConv2d
        self.conv1 = QuantizedConv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.conv2 = QuantizedConv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.conv3 = QuantizedConv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.conv4 = QuantizedConv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE)
        
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        # 256 Filter * 2 * 2 = 1024
        self.fc_input_dim = 256 * 2 * 2 

        # Ersetze nn.Linear durch QuantizedLinear
        self.fc1 = QuantizedLinear(self.fc_input_dim, 1024)
        self.fc2 = QuantizedLinear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Der Forward-Pass bleibt fast gleich, nutzt aber nun die Quantized-Logik
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # (Batch, 256, H, W) -> (Batch, 256, 1, 1)
        x = self.global_pool(x)
        
        # (Batch, 256)
        x = torch.flatten(x, 1)
        
        # Klassifizierung
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# class CNN(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE)
#         self.dropout = nn.Dropout(0.5)

#         self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

#         # 256 Filter * 2 * 2 = 1024
#         self.fc_input_dim = 256 * 2 * 2 

#         self.fc1 = nn.Linear(self.fc_input_dim, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))

#         # (Batch, 256, H, W) -> (Batch, 256, 1, 1)
#         x = self.global_pool(x)
        
#         # (Batch, 256)
#         x = torch.flatten(x, 1)
        
#         # Klassifizierung
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# 22.12.2025: Accuracy at 84%