import torch
import torch.nn as nn
import torch.nn.functional as F
# from copy import deepcopy

from src.utility.quantizer import Quantization
from src.utility.logging import get_logger
from src.utility.config import QUANTIZATION_METHOD, QUANTIZATION_NUM_BITS, QUANTIZATION_NUM_BATCHES

logger = get_logger(__name__)

class QuantizedLayers:
    def _init_quantized_buffers(self):
        # weight
        self.register_buffer('weight_int8', None)
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.is_quantized = False
        # activation storage
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0))
        self.activation_calibrated = False
        self.quant_method = QUANTIZATION_METHOD
        self.num_bits = QUANTIZATION_NUM_BITS

    def quantized_storage(self, num_bits=QUANTIZATION_NUM_BITS, method=QUANTIZATION_METHOD):

        if not hasattr(self, 'weight') or self.weight is None:
            return

        logger.debug(f"Quantizing layer (storage): {method} {num_bits}-bit")

        _, w_int, scale, zp = Quantization.quantize_tensor(self.weight.data, method=method, num_bits=num_bits)

        self.register_buffer('weight_int', w_int)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zp)
        self.is_quantized = True
        self.quant_method = method
        self.num_bits = num_bits

        try:
            del self.weight
        except Exception:
            pass

        self.register_parameter('weight', None)
          
class QuantizedLinear(nn.Linear, QuantizedLayers):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)
        self._init_quantized_buffers()

    def forward(self, input):
        if self.is_quantized and self.activation_calibrated:
            input, _ =Quantization.quantize_with_params(
                input, self.act_scale, self.act_zero_point,
                method=self.quant_method, num_bits=self.num_bits
                )

        if self.is_quantized:
            w_dequant = Quantization.dequantize_from_int(
                self.weight_int, self.scale, 
                self.zero_point, method=self.quant_method
                )
            return F.linear(input, w_dequant, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

class QuantizedConv2d(nn.Conv2d, QuantizedLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quantized_buffers()

    def forward(self, input):
        if self.is_quantized and self.activation_calibrated:
            input, _ = Quantization.quantize_with_params(
                input, self.act_scale, self.act_zero_point, 
                method=self.quant_method, num_bits=self.num_bits
                )

        if self.is_quantized:
            w_dequant = Quantization.dequantize_from_int(
                self.weight_int, self.scale, 
                self.zero_point, method=self.quant_method
                )
            
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), 
                                w_dequant, self.bias, self.stride, 0, self.dilation, self.groups)
            return F.conv2d(input, w_dequant, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super().forward(input)

def replace_layers_with_quantizable(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            hasBias = module.bias is not None
            new_layer = QuantizedLinear(module.in_features, module.out_features, bias=hasBias)
            new_layer.weight.data = module.weight.data.clone()

            if hasBias:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)

        elif isinstance(module, nn.Conv2d):
            hasBias = module.bias is not None
            new_layer = QuantizedConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation, module.groups,
                bias=hasBias, padding_mode=module.padding_mode
            )
            new_layer.weight.data = module.weight.data.clone()
            if hasBias:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            replace_layers_with_quantizable(module)
    return model

def calibrated_model_activation(model, data_loader, device='cpu', num_batches=QUANTIZATION_NUM_BATCHES, method=QUANTIZATION_METHOD, num_bits=QUANTIZATION_NUM_BITS):
    model.eval()
    model.to(device)

    # Dictionary to store min and max for each layer
    active_stats = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if output is None:
                return
            
            # Track both Min and Max for proper Affine support
            curr_min = output.min().item()
            curr_max = output.max().item()
            
            if name not in active_stats:
                active_stats[name] = {'min': curr_min, 'max': curr_max}
            else:
                active_stats[name]['min'] = min(active_stats[name]['min'], curr_min)
                active_stats[name]['max'] = max(active_stats[name]['max'], curr_max)
        return hook
    
    for name,module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLayers)):
            handles.append(module.register_forward_hook(make_hook(name)))

    logger.info(f"Calibrating with {num_batches} batches...")
    batches = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            model(data)
            batches += 1
            if batches >= num_batches:
                break

    for h in handles:
        h.remove()

    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLayers)):
            stats = active_stats.get(name)

            if stats is None:
                continue
            x_min = stats['min']
            x_max = stats['max']

            # Affine: Use Min and Max
            if method == 'affine':
                q_min = 0
                q_max = 2**num_bits - 1
                if x_max == x_min:
                    scale = 1.0
                else:
                    scale = (x_max - x_min) / float(q_max - q_min)
                
                module.act_scale = torch.tensor(scale)
                module.act_zero_point = torch.round(torch.tensor(q_min - (x_min / scale)))
            
            # Symmetric: Use Max(Abs)
            else:
                q_max = (2 ** (num_bits - 1)) - 1
                abs_max = max(abs(x_min), abs(x_max))
                
                if abs_max == 0:
                    scale = 1.0
                else:
                    scale = abs_max / float(q_max)
                    
                module.act_scale = torch.tensor(scale)
                module.act_zero_point = torch.tensor(0.0)

            module.activation_calibrated = True
            module.num_bits = num_bits
            module.quant_method = method

    logger.info("Activation calibration finished for quantizable layers")
    return model
        