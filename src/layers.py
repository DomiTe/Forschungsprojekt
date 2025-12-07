import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.quantizer import Quantization
from src.logging import get_logger

logger = get_logger(__name__)

class QuantizedLayers:
    def _init_quantized_buffers(self):
        # weight
        self.register_buffer('weight_int8', None)
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.is_quantized = False

    def quantized_storage(self, num_bits=8, method='symmetric'):

        if self.weight is None: return

        logger.debug(f"Quantizing layer (storage): {method}{num_bits}-bit")

        if method == 'affine':
            _, w_int, scale, zp = Quantization.affine_quantization(self.weight.data, num_bits)
        else:
            _, w_int, scale, zp = Quantization.symmetric_quantization(self.weight.data, num_bits)

        self.weight_int8 = w_int
        self.scale = scale
        self.zero_point = zp
        self.is_quantized = True

        del self.weight
        self.register_parameter('weight', None)

class QuantizedLinear(nn.Linear, QuantizedLayers):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)
        self._init_quantized_buffers()

    def forward(self, input):
        if self.is_quantized:
            w_dequant = (self.weight_int8.float() - self.zero_point) * self.scale
            return F.linear(input, w_dequant, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

class QuantizedConv2d(nn.Conv2d, QuantizedLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quantized_buffers()

    def forward(self, input):
        if self.is_quantized:
            w_dequant = (self.weight_int8.float() - self.zero_point) * self.scale
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), 
                                w_dequant, self.bias, self.stride, 0, self.dilation, self.groups)
            return F.conv2d(input, w_dequant, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super().forward(input)

def replace_layers_with_quantizable(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            hasBias = module.bias is not None
            new_layer = QuantizedLinear(module.in_features, module.out_features, bias=hasBias)
            new_layer.weight.data = module.weight.data

            if hasBias:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Conv2d):
            hasBias = module.bias is not None
            new_layer = QuantizedConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation, module.groups,
                bias=hasBias, padding_mode=module.padding_mode
            )
            new_layer.weight.data = module.weight.data

            if hasBias:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)
        else:
            replace_layers_with_quantizable(module)
    return model
        
        