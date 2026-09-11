import torch
import torch.nn as nn


class TrueInt8Linear(nn.Module):
    """A physical INT8 linear layer backed by Triton."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale_w", torch.empty((out_features,), dtype=torch.float32))
        
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        pass

def convert_qat_to_real_int8(module: nn.Module, min_exp: int = -8, max_exp: int = 1):
    """
    Recursively replaces QAT layers with True INT8 hardware layers.
    """
    for name, child in module.named_children():
        if type(child).__name__ == "QLinear": 
            with torch.no_grad():
                w_fp32 = child.weight.data
                x_safe = torch.where(w_fp32 == 0, torch.tensor(1e-9, device=w_fp32.device), w_fp32)
                sign = torch.sign(x_safe)
                log2_val = torch.round(torch.log2(torch.abs(x_safe)))
                clamped_exp = torch.clamp(log2_val, min_exp, max_exp)
                w_int8 = (sign * (2.0 ** clamped_exp)).to(torch.int8)
            
            real_int8_layer = TrueInt8Linear(child.in_features, child.out_features, bias=(child.bias is not None))
            real_int8_layer.weight_int8.copy_(w_int8)
            
            
            if child.bias is not None:
                real_int8_layer.bias.copy_(child.bias.data)

            setattr(module, name, real_int8_layer)
            
        else:
            convert_qat_to_real_int8(child, min_exp, max_exp)