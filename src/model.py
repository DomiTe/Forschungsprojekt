import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utility.config import KERNEL_SIZE, STRIDE, CHANNELS, NUM_CLASSES
from src.layers import QuantizedConv2d, QuantizedLinear, QuantizedLayerMixin

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        
        # Block 1
        self.conv1 = QuantizedConv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.conv2 = QuantizedConv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        # Block 3
        self.conv3 = QuantizedConv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        # Block 4
        self.conv4 = QuantizedConv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc_input_dim = 256 * 2 * 2 
        
        # Block 5 (FC) - No BN needed here based on last experiment
        self.fc1 = QuantizedLinear(self.fc_input_dim, 1024, bias=True)
        self.relu5 = nn.ReLU()

        self.fc2 = QuantizedLinear(1024, num_classes, bias=True)

    def forward(self, x):
        # Note: Input x is fake-quantized inside conv1's forward
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        x = self.relu5(self.fc1(x))
        x = self.fc2(x) # No ReLU on final logits
        return x

        # --- Helfer-Methoden für das Experiment ---

    def convert_to_quantized(self, method='symmetric', bits=8):
        """Aktiviert Quantisierung für alle Layer"""
        for module in self.modules():
            if isinstance(module, QuantizedLayerMixin):
                module.prepare_quantization(method, bits)

    def convert_to_baseline(self):
        """Deaktiviert Quantisierung (Reset auf Float32)"""
        for module in self.modules():
            if isinstance(module, QuantizedLayerMixin):
                module.disable_quantization()

    def permanently_quantize_weights(self):
        """Wandelt alle Gewichte im Modell permanent in INT8 um (spart Speicher)."""
        for module in self.modules():
            if isinstance(module, QuantizedLayerMixin):
                module.convert_weights_to_int8()