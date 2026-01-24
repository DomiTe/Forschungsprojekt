import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

class SeminarAffineObserver(MinMaxObserver):
    """
    [cite_start]Implements Affine Quantization from Topic 4, Slide 16[cite: 126].
    
    Formula:
      S = (x_max - x_min) / (q_max - q_min)
      Z = round(q_min - x_min / S)
    """
    def calculate_qparams(self):
        # 1. Get min/max from the observed data
        min_val, max_val = self.min_val, self.max_val
        
        # Guard against empty tensors
        if min_val == float("inf") or max_val == float("-inf"):
            return torch.tensor([1.0]), torch.tensor([0])
            
        # 2. Get q_min and q_max (e.g., -128 and 127)
        q_min, q_max = self.quant_min, self.quant_max
        
        # 3. Calculate Scale (S)
        # S = (x_max - x_min) / (q_max - q_min)
        numerator = max_val - min_val
        denominator = float(q_max - q_min)
        
        scale = numerator / denominator
        scale = torch.max(scale, torch.tensor(1e-6)) # Avoid division by zero
        
        # 4. Calculate Zero Point (Z)
        # Z = round(q_min - x_min / S)
        zero_point = q_min - (min_val / scale)
        zero_point = torch.round(zero_point)
        
        # Clamp Z to valid int8 range so PyTorch doesn't crash
        zero_point = torch.clamp(zero_point, q_min, q_max)
        
        return scale, zero_point.int()

# ==========================================
# 1. SYMMETRIC ACTIVATION (The Fix)
# ==========================================
class SeminarSymmetricActivationObserver(MinMaxObserver):
    """
    For ACTIVATIONS (Per-Tensor, Unsigned quint8).
    Adapts the Seminar Formula to the 0-255 range.
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val == float("inf") or max_val == float("-inf"):
            return torch.tensor([1.0]), torch.tensor([0])

        # Formula: Use 254 steps (Seminar constraint) but map to [0, 255]
        denominator = 254.0 
        
        # 1. Scale
        scale = (max_val - min_val) / denominator
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # 2. Zero Point (Corrected for Unsigned quint8)
        # Standard Formula: Z = q_min - (x_min / S)
        # For quint8, q_min is 0.
        zero_point = 0.0 - (min_val / scale)
        
        # Clamp to valid quint8 range [0, 255]
        zero_point = torch.clamp(torch.round(zero_point), 0, 255)
        
        return scale, zero_point.int()

# ==========================================
# 2. SYMMETRIC WEIGHT (Keep as is - it works for Signed)
# ==========================================
class SeminarSymmetricWeightObserver(PerChannelMinMaxObserver):
    """
    For WEIGHTS (Per-Channel, Signed qint8).
    Strictly follows Seminar Formula mapping to [-127, 127].
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
             return torch.tensor([1.0]), torch.tensor([0])

        denominator = 254.0 
        scale = (max_val - min_val) / denominator
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # Formula: Z = -127 - (x_min / S)
        # This is correct for qint8 (Signed)
        zero_point = -127.0 - (min_val / scale)
        zero_point = torch.clamp(torch.round(zero_point), -128, 127)
        
        return scale, zero_point.int()

# ==========================================
# 3. PoT OBSERVERS (Ensure similar logic)
# ==========================================
class SeminarPoTActivationObserver(MinMaxObserver):
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val == float("inf") or max_val == float("-inf"):
            return torch.tensor([1.0]), torch.tensor([0])
            
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        # Map to 127 steps (half of 254)
        scale = max_abs / 127.0
        scale = torch.max(scale, torch.tensor(1e-6))
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
        
        # For PoT/Symmetric Unsigned (0-255), if data is ReLU (0 to Max), 
        # the Zero Point 0 is correct.
        return scale, torch.zeros(1).int()

class SeminarPoTWeightObserver(PerChannelMinMaxObserver):
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
             return torch.tensor([1.0]), torch.tensor([0])

        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_abs / 127.0
        scale = torch.max(scale, torch.tensor(1e-6))
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
        
        return scale, torch.zeros_like(scale).int()