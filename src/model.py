import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utility.config import KERNEL_SIZE, STRIDE, CHANNELS, NUM_CLASSES
from src.layers import QuantizedConv2d, QuantizedLinear, QuantizedLayerMixin

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        
        # standarised conv2d layers
        self.conv1 = nn.Conv2d(CHANNELS, 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)
        
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Calculation of dimension: 256 Channel * 2x2 Pooling
        self.fc_input_dim = 256 * 2 * 2 

        # standarised linear layers
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Normaler Forward-Pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

    # --- Helfer-Methoden für das Experiment ---

    # def convert_to_quantized(self, method='symmetric', bits=8):
    #     """Aktiviert Quantisierung für alle Layer"""
    #     for module in self.modules():
    #         if isinstance(module, QuantizedLayerMixin):
    #             module.prepare_quantization(method, bits)

    # def convert_to_baseline(self):
    #     """Deaktiviert Quantisierung (Reset auf Float32)"""
    #     for module in self.modules():
    #         if isinstance(module, QuantizedLayerMixin):
    #             module.disable_quantization()

    # def permanently_quantize_weights(self):
    #     """Wandelt alle Gewichte im Modell permanent in INT8 um (spart Speicher)."""
    #     for module in self.modules():
    #         if isinstance(module, QuantizedLayerMixin):
    #             module.convert_weights_to_int8()