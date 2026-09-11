import torch
import torch.nn as nn

def export_po2_to_int8(po2_weights: torch.Tensor, min_exp: int, max_exp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts the QAT floating-point weights into deployment-ready INT8."""
    with torch.no_grad():
        x_safe = torch.where(po2_weights == 0, torch.tensor(1e-9, device=po2_weights.device), po2_weights)
        sign = torch.sign(x_safe)
        log2_val = torch.round(torch.log2(torch.abs(x_safe)))
        clamped_exp = torch.clamp(log2_val, min_exp, max_exp)
        
        w_int = (sign * (2.0 ** clamped_exp)).to(torch.int8)
        
        scale_w = torch.ones(w_int.shape[0], device=po2_weights.device, dtype=torch.float32)
        
        return w_int, scale_w