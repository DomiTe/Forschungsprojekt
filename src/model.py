import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
from src.utility.config import KERNEL_SIZE, STRIDE, CHANNELS, NUM_CLASSES
# from src.layers import QuantizedConv2d, QuantizedLinear, QuantizedLayerMixin

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        
        self.quant = QuantStub()

        self.conv1 = nn.conv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = nn.Linear(256 * 2 * 2 , 1024)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dequant =DeQuantStub()


    def forward(self, x):

        x = self.quant()

        x = F.relu1(self.conv1(x))
        x = F.relu2(self.conv2(x))
        x = F.relu3(self.conv3(x))
        x = F.relu4(self.conv4(x))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = F.relu5(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant
        
        return x