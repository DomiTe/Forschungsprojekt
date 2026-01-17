import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utility.quantizer import Quantization

def get_multiplier_shift(effective_scale):
    if effective_scale == 0: return 0, 0
    m, e = math.frexp(effective_scale)
    shift = 31 - e
    multiplier = int(round(m * (1 << 31)))
    return multiplier, shift

class QuantizedLayerMixin:
    def _init_quantization_attributes(self):
        self.quant_mode = False 
        self.activation_calibrated = False
        self.int_mode = False
        
        # Weights
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0)) 
        self.register_buffer('out_scale', torch.tensor(1.0))
        self.register_buffer('out_zero_point', torch.tensor(0.0))
        
        self.quant_method = 'symmetric'
        self.num_bits = 8

        # Cache variables to avoid .item() calls during inference
        self._cached_shift = None
        self._cached_pad = None
        self._cached_out_pad = None

    def prepare_quantization(self, method='symmetric', bits=8):
        self.quant_method = 'symmetric'
        self.num_bits = bits
        _, _, scale, zp = Quantization.symmetric_quantization(self.weight, num_bits=bits)
        self.weight_scale.copy_(scale)
        self.weight_zero_point.copy_(zp)
        self.quant_mode = True

    def convert_weights_to_int8(self):
        if not self.quant_mode: return
        _, w_int = Quantization.quantize_with_params(
            self.weight, self.weight_scale, self.weight_zero_point, 
            'symmetric', self.num_bits
        )
        self.weight = nn.Parameter(w_int.to(torch.int8), requires_grad=False)

    def disable_quantization(self):
        """Disables quantization and resets to float32 mode."""
        self.quant_mode = False
        self.int_mode = False


class QuantizedConv2d(nn.Conv2d, QuantizedLayerMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self._init_quantization_attributes()
        
        self.register_buffer('output_multiplier', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('output_shift', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('pad_amount', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('weight_int8_padded', None)

    def prepare_integer_inference(self):
        if not self.activation_calibrated: return

        # 1. Calc Multiplier
        real_input_scale = self.act_scale 
        real_weight_scale = self.weight_scale
        real_output_scale = self.out_scale 
        effective_scale = (real_input_scale * real_weight_scale) / real_output_scale
        mult, shift = get_multiplier_shift(effective_scale.item())
        
        self.output_multiplier.fill_(mult)
        self.output_shift.fill_(shift)

        # 2. Weights
        self.convert_weights_to_int8()
        w_flat = self.weight.view(self.out_channels, -1)
        k_dim = w_flat.shape[1]
        
        # 3. Alignment
        remainder = k_dim % 8
        if remainder != 0:
            pad = 8 - remainder
            w_padded = F.pad(w_flat, (0, pad), "constant", 0)
            self.pad_amount.fill_(pad)
            self.register_buffer('weight_int8_padded', w_padded)
        else:
            self.pad_amount.fill_(0)
            self.register_buffer('weight_int8_padded', w_flat)

        # 4. Bias
        if self.bias is not None:
             bias_scale = real_input_scale * real_weight_scale
             bias_int32 = torch.round(self.bias.detach() / bias_scale).to(torch.int32)
             self.register_buffer('bias_int32', bias_int32)
             
        self.int_mode = True
        
        # Pre-cache values to be safe
        self._cached_shift = shift
        self._cached_pad = self.pad_amount.item()

    def forward(self, x):
        if not self.quant_mode: return super(QuantizedConv2d, self).forward(x)
            
        if not self.int_mode: 
            w_fake, _ = Quantization.quantize_with_params(self.weight, self.weight_scale, self.weight_zero_point, 'symmetric', self.num_bits)
            x_fake, _ = Quantization.quantize_with_params(x, self.act_scale, self.act_zero_point, 'symmetric', self.num_bits)
            return F.conv2d(x_fake, w_fake, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # --- OPTIMIZED INT INFERENCE ---
        
        # 1. Cache Fetching (No .item() in loop!)
        if self._cached_shift is None:
            self._cached_shift = self.output_shift.item()
            self._cached_pad = self.pad_amount.item()

        shift = self._cached_shift
        pad = self._cached_pad

        # 2. Quantize Input
        x_int8 = torch.round(x / self.act_scale).clamp(-128, 127).to(torch.int8)

        # 3. Unfold (Float16)
        x_unfolded = F.unfold(x_int8.half(), self.kernel_size, self.dilation, self.padding, self.stride)
        del x_int8
        x_unfolded = x_unfolded.to(torch.int8)
        
        N, C_tot, L = x_unfolded.shape
        x_gemm = x_unfolded.transpose(1, 2).reshape(N * L, C_tot)
        del x_unfolded
        
        # 4. Dynamic Padding
        if pad > 0:
            x_gemm = F.pad(x_gemm, (0, pad), "constant", 0)

        # 5. Matrix Mult
        if self.weight_int8_padded is not None:
             res_int32 = torch._int_mm(x_gemm, self.weight_int8_padded.t())
        else:
             res_int32 = torch._int_mm(x_gemm, self.weight.view(self.out_channels, -1).t())
        del x_gemm

        if hasattr(self, 'bias_int32'):
            res_int32 += self.bias_int32
            
        # 6. Rescale + Rounding
        acc_long = res_int32.to(torch.int64)
        del res_int32
        
        acc_long *= self.output_multiplier
        
        # Rounding Correction (Bitwise math on CPU int 'shift' is fast)
        if shift > 0:
            acc_long += (1 << (shift - 1))
            acc_long >>= shift
        
        x_out_int8 = acc_long.to(torch.int32).clamp(-128, 127).to(torch.int8)
        del acc_long
        
        # 7. Reshape Output
        out_h = int((x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        out_w = int((x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x_out_int8 = x_out_int8.view(N, out_h, out_w, self.out_channels).permute(0, 3, 1, 2)

        return x_out_int8.float() * self.out_scale


class QuantizedLinear(nn.Linear, QuantizedLayerMixin):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias=bias)
        self._init_quantization_attributes()
        
        self.register_buffer('output_multiplier', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('output_shift', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('pad_output', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('weight_int8_padded', None)
        self.register_buffer('bias_int32_padded', None)

    def prepare_integer_inference(self):
        if not self.activation_calibrated: return
        
        real_input_scale = self.act_scale 
        real_weight_scale = self.weight_scale
        real_output_scale = self.out_scale 
        effective_scale = (real_input_scale * real_weight_scale) / real_output_scale
        mult, shift = get_multiplier_shift(effective_scale.item())
        
        self.output_multiplier.fill_(mult)
        self.output_shift.fill_(shift)
        
        self.convert_weights_to_int8()
        
        bias_int32 = None
        if self.bias is not None:
             bias_scale = real_input_scale * real_weight_scale
             bias_int32 = torch.round(self.bias.detach() / bias_scale).to(torch.int32)
             self.register_buffer('bias_int32', bias_int32)
        
        # Alignment
        out_f = self.out_features
        remainder = out_f % 8
        if remainder != 0:
            pad = 8 - remainder
            w_padded = F.pad(self.weight, (0, 0, 0, pad), "constant", 0)
            self.register_buffer('weight_int8_padded', w_padded)
            self.pad_output.fill_(pad)
            
            if bias_int32 is not None:
                b_padded = F.pad(bias_int32, (0, pad), "constant", 0)
                self.register_buffer('bias_int32_padded', b_padded)
        else:
            self.pad_output.fill_(0)
            self.register_buffer('weight_int8_padded', self.weight)
            self.register_buffer('bias_int32_padded', bias_int32)
        
        self.int_mode = True
        
        # Cache
        self._cached_shift = shift
        self._cached_out_pad = self.pad_output.item()

    def forward(self, x):
        if not self.quant_mode: return super(QuantizedLinear, self).forward(x)
        
        if not self.int_mode:
             w_fake, _ = Quantization.quantize_with_params(self.weight, self.weight_scale, self.weight_zero_point, 'symmetric', self.num_bits)
             x_fake, _ = Quantization.quantize_with_params(x, self.act_scale, self.act_zero_point, 'symmetric', self.num_bits)
             return F.linear(x_fake, w_fake, self.bias)

        # Cache Fetch
        if self._cached_shift is None:
            self._cached_shift = self.output_shift.item()
            self._cached_out_pad = self.pad_output.item()
        
        shift = self._cached_shift
        pad = self._cached_out_pad

        x_int8 = torch.round(x / self.act_scale).clamp(-128, 127).to(torch.int8)
        
        w_used = self.weight_int8_padded if self.weight_int8_padded is not None else self.weight
        acc_int32 = torch._int_mm(x_int8, w_used.to(torch.int8).t())
        
        b_used = self.bias_int32_padded if self.bias_int32_padded is not None else getattr(self, 'bias_int32', None)
        if b_used is not None:
            acc_int32 += b_used
            
        acc_long = acc_int32.to(torch.int64)
        del acc_int32
        
        acc_long *= self.output_multiplier
        
        if shift > 0:
            acc_long += (1 << (shift - 1))
            acc_long >>= shift
        
        x_out_int8 = acc_long.to(torch.int32).clamp(-128, 127).to(torch.int8)
        
        if pad > 0:
            x_out_int8 = x_out_int8[:, :self.out_features]
        
        return x_out_int8.float() * self.out_scale