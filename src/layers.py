import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utility.quantizer import Quantization
import logging

logger = logging.getLogger(__name__)

class QuantizedLayerMixin:
    """
    Diese Klasse fügt Quantisierungs-Fähigkeiten zu Standard-Layern hinzu.
    Sie hält den State (Scale, ZeroPoint, Mode).
    """
    def _init_quantization_attributes(self):
        # Default: Quantisierung ist AUS (Verhält sich wie Baseline/Float)
        self.quant_mode = False 
        self.activation_calibrated = False
        
        # Parameter für Gewichts-Quantisierung
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        
        # Parameter für Aktivierungs-Quantisierung (für später)
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0))
        
        self.quant_method = 'symmetric'
        self.num_bits = 8

    def prepare_quantization(self, method='symmetric', bits=8):
        """
        Berechnet Scale & Zero-Point basierend auf den aktuellen Float-Gewichten
        und aktiviert den Quantisierungs-Modus.
        """
        self.quant_method = method
        self.num_bits = bits
        
        # Determine reduction dimensions for per-channel quantization
        if isinstance(self, nn.Conv2d):
            # Conv2d weights: (C_out, C_in, kH, kW)
            w_dim = (1, 2, 3)
        elif isinstance(self, nn.Linear):
            # Linear weights: (out_features, in_features)
            w_dim = 1
        else:
            w_dim = None
            
        # Berechne Parameter für die Gewichte (W)
        if method == 'affine':
            _, _, scale, zp = Quantization.affine_quantization(self.weight, num_bits=bits, dim=w_dim)
        elif method == 'power2':
             _, _, scale, zp = Quantization.power_of_two_quantization(self.weight, num_bits=bits, dim=w_dim)
        else: # symmetric
            _, _, scale, zp = Quantization.symmetric_quantization(self.weight, num_bits=bits, dim=w_dim)
        
        self.weight_scale.data = scale.clone().detach()
        self.weight_zero_point.data = zp.clone().detach()
        # self.weight_scale.copy_(scale)
        # self.weight_zero_point.copy_(zp)
        self.quant_mode = True

        logger.debug(f"Layer prepared: {method} ({bits} bit). Scale shape={scale.shape}")
    
    def convert_weights_to_int8(self):
        """
        Wandelt die gespeicherten Float-Gewichte physisch in INT8/UINT8 um.
        Dies reduziert die Modellgröße beim Speichern drastisch.
        """
        if not self.quant_mode:
            return

        # 1. Berechne die Integer-Werte
        _, w_int = Quantization.quantize_with_params(
            self.weight, 
            self.weight_scale, 
            self.weight_zero_point, 
            self.quant_method, 
            self.num_bits
        )

        # 2. Wähle den Datentyp (Affine = uint8, Symmetrisch = int8)
        if self.quant_method == 'affine':
            target_dtype = torch.uint8
        else:
            target_dtype = torch.int8

        # 3. ERSETZE den Parameter 'self.weight'
        # Wir löschen den Float-Tensor und setzen einen neuen Int-Parameter
        self.weight = nn.Parameter(w_int.to(target_dtype), requires_grad=False)
        
        logger.debug(f"Gewichte konvertiert zu {target_dtype}")

    def disable_quantization(self):
        self.quant_mode = False


class QuantizedConv2d(nn.Conv2d, QuantizedLayerMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(QuantizedConv2d, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=padding,
            bias=bias
            )
        self._init_quantization_attributes()

    def forward(self, x):
        if not self.quant_mode:
            return super(QuantizedConv2d, self).forward(x)

        if self.weight.dtype in [torch.int8, torch.uint8]:
            w_fake = (self.weight.float() - self.weight_zero_point) * self.weight_scale
        else:
            w_fake, _ = Quantization.quantize_with_params(
                self.weight, 
                self.weight_scale, 
                self.weight_zero_point, 
                self.quant_method, 
                self.num_bits
            )

        x_fake = x
        if self.activation_calibrated:
            x_fake, _ = Quantization.quantize_with_params(
                x, self.act_scale, self.act_zero_point, 
                self.quant_method, self.num_bits
                )
        
        return F.conv2d(x_fake, w_fake, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedLinear(nn.Linear, QuantizedLayerMixin):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__(
            in_features, 
            out_features, 
            bias=bias
        )
        self._init_quantization_attributes()

    def forward(self, x):
        if not self.quant_mode:
            return super(QuantizedLinear, self).forward(x)
        
        if self.weight.dtype in [torch.int8, torch.uint8]:
            w_fake = (self.weight.float() - self.weight_zero_point) * self.weight_scale
        else:
            w_fake, _ = Quantization.quantize_with_params(
                self.weight, 
                self.weight_scale, 
                self.weight_zero_point, 
                self.quant_method, 
                self.num_bits
            )

        x_fake = x
        if self.activation_calibrated:
            x_fake, _ = Quantization.quantize_with_params(
                x, self.act_scale, self.act_zero_point, 
                self.quant_method, self.num_bits
                )
        
        return F.linear(x_fake, w_fake, self.bias)