import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utility.quantizer import Quantization

class QuantizedLayerMixin:
    def _init_quantization_attributes(self):
        self.quant_mode = False 
        self.activation_calibrated = False
        self.int_mode = False
        
        # Buffers for scales
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0)) 
        self.register_buffer('out_scale', torch.tensor(1.0))
        self.register_buffer('out_zero_point', torch.tensor(0.0))
        
        self.quant_method = 'symmetric'
        self.num_bits = 8
        self.packed_params = None # CPU specific storage

    def prepare_quantization(self, method='symmetric', bits=8):
        self.quant_method = 'symmetric'
        self.num_bits = bits
        _, _, scale, zp = Quantization.symmetric_quantization(self.weight, num_bits=bits)
        self.weight_scale.copy_(scale)
        self.weight_zero_point.copy_(zp)
        self.quant_mode = True

    def disable_quantization(self):
        self.quant_mode = False
        self.int_mode = False

    def get_quantized_state(self):
        """Returns the Int8 weights and scales for saving."""
        return {
            'weight_int8': self.weight_int8,
            'weight_scale': self.weight_scale,
            'act_scale': self.act_scale,
            'out_scale': self.out_scale
        }
    def load_quantized_state(self, state_dict):
        """Restores the Int8 state and re-packs for the hardware."""
        self.weight_int8 = state_dict['weight_int8']
        self.weight_scale = state_dict['weight_scale']
        self.act_scale = state_dict['act_scale']
        self.out_scale = state_dict['out_scale']

class QuantizedConv2d(nn.Conv2d, QuantizedLayerMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self._init_quantization_attributes()

    def prepare_integer_inference(self):
        if not self.activation_calibrated: return

        # 1. Weights: Quantize to qint8 (Symmetric)
        w_float = self.weight.detach().cpu().float()
        qweight = torch.quantize_per_tensor(
            w_float, self.weight_scale.item(), 0, torch.qint8
        )
        
        # 2. Correct Argument Order for conv2d_prepack:
        # (weight, bias, stride, padding, dilation, groups)
        self.packed_params = torch.ops.quantized.conv2d_prepack(
            qweight, 
            self.bias,       # Bias must be the second argument
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        
        # 3. Save as int8 buffer for "Research Size" demonstration
        # This extract raw bytes from the quantized tensor
        self.register_buffer('weight_int8', qweight.int_repr())
        
        # 4. Clear float weights to simulate compression
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        # We leave bias as is because the packer holds a reference to it
        
        self.int_mode = True

    def forward(self, x):
        if not self.quant_mode: return super(QuantizedConv2d, self).forward(x)
        if not self.int_mode: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

        # 1. Quantize Input using the calibrated Midpoint (128)
        # This preserves both positive and negative signals
        x_q = torch.quantize_per_tensor(
            x, 
            self.act_scale.item(), 
            int(self.act_zero_point.item()), # This will be 128
            torch.quint8
        )
        
        # 2. Execute the FBGEMM Optimized Kernel
        # We pass the out_scale and out_zero_point (128) to the C++ backend
        out_q = torch.ops.quantized.conv2d(
            x_q, 
            self.packed_params, 
            self.out_scale.item(), 
            int(self.out_zero_point.item())
        )
        
        return out_q.dequantize()

class QuantizedLinear(nn.Linear, QuantizedLayerMixin):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias=bias)
        self._init_quantization_attributes()

    def prepare_integer_inference(self):
        if not self.activation_calibrated: return

        w_float = self.weight.detach().cpu().float()
        qweight = torch.quantize_per_tensor(
            w_float, self.weight_scale.item(), 0, torch.qint8
        )
        
        # Correct Argument Order for linear_prepack: (weight, bias)
        self.packed_params = torch.ops.quantized.linear_prepack(qweight, self.bias)
        
        self.register_buffer('weight_int8', qweight.int_repr())
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        
        self.int_mode = True

    def forward(self, x):
        if not self.quant_mode: return super(QuantizedLinear, self).forward(x)
        if not self.int_mode: return F.linear(x, self.weight, self.bias)

        # Apply same midpoint logic for Linear layers
        x_q = torch.quantize_per_tensor(
            x, 
            self.act_scale.item(), 
            int(self.act_zero_point.item()), 
            torch.quint8
        )
        
        out_q = torch.ops.quantized.linear(
            x_q, 
            self.packed_params, 
            self.out_scale.item(), 
            int(self.out_zero_point.item())
        )
        
        return out_q.dequantize()