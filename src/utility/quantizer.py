import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Quantization:
    """
    Core Fake Quantization logic.
    Simulates INT8 behavior using Float32 math (Fake Quantization).
    """

    @staticmethod
    def _core_quantization(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, q_min: int, q_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Quantize: (x / S) + Z
        2. Clamp: [q_min, q_max]
        3. De-quantize: (x_q - Z) * S
        """
        # 1. Quantize
        # Add epsilon to scale to prevent division by zero
        scale = scale + 1e-6
        x_int = torch.round((tensor / scale) + zero_point)
        
        # 2. Clamp
        x_int = torch.clamp(x_int, q_min, q_max)
        
        # 3. De-Quantize
        x_dequant = (x_int - zero_point) * scale

        return x_dequant, x_int
    
    @staticmethod
    def _get_min_max(tensor: torch.Tensor, dim=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates min/max dynamically based on channel dimensions.
        """  
        if dim is None:
            # Standard Per-Tensor
            return tensor.min(), tensor.max()
        else:
            # Per-Channel: keepdim=True ensures the shape stays (C, 1, 1, 1) 
            # so it broadcasts correctly when dividing the weights later.
            t_min = torch.amin(tensor, dim=dim, keepdim=True)
            t_max = torch.amax(tensor, dim=dim, keepdim=True)
            return t_min, t_max
        
    @staticmethod
    def affine_quantization(tensor: torch.Tensor, num_bits: int = 8, dim=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asymmetric (Affine). Maps [min, max] -> [0, 2^b - 1].
        """
        t_min, t_max = Quantization._get_min_max(tensor, dim=dim)        
        
        q_min, q_max = 0, (2 ** num_bits) - 1

        scale = (t_max - t_min) / float(q_max - q_min)
        scale = torch.max(scale, torch.tensor(1e-6))

        zero_point = q_min - (t_min / scale)
        zero_point = torch.clamp(torch.round(zero_point), q_min, q_max)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def symmetric_quantization(tensor: torch.Tensor, num_bits: int = 8, dim=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symmetric Quantization.
        - Weights: Signed [-127, 127], ZP=0
        - Activations (if positive): Unsigned [0, 255], ZP=0
        """
        t_min, t_max = Quantization._get_min_max(tensor, dim=dim)        

        q_max = (2 ** (num_bits - 1)) - 1  # 127
        q_min = -q_max                     #- 127

        t_abs_max = torch.max(torch.abs(t_min), torch.abs(t_max))
        
        scale = t_abs_max / float(q_max)
        zero_point = torch.tensor(0.0).to(tensor.device)
        
        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def power_of_two_quantization(tensor: torch.Tensor, num_bits: int = 8, dim=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power-of-Two (PoT).
        Same limits as Symmetric, but Scale is rounded to nearest 2^k.
        """
        t_min, t_max = Quantization._get_min_max(tensor, dim=dim)        

        q_max = (2 ** (num_bits - 1)) - 1  # 127
        q_min = -q_max                     #- 127

        t_abs_max = torch.max(torch.abs(t_min), torch.abs(t_max))
        
        scale_ideal = t_abs_max / float(q_max)
        
        # Force Scale to 2^k
        # log2(0) protection
        scale_ideal = torch.max(scale_ideal, torch.tensor(1e-6))
        scale = 2 ** torch.round(torch.log2(scale_ideal))
        
        zero_point = torch.tensor(0.0).to(tensor.device)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point
    
    @staticmethod
    def quantize_with_params(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, method: str, num_bits: int = 8):
        """
        Wendet Quantisierung mit BEREITS BEKANNTEN Parametern an.
        Wird im Forward-Pass der Layer benutzt (Inference).
        """
        # Bestimme Limits basierend auf der Methode
        if method == 'affine':
            q_min = 0
            q_max = (2 ** num_bits) - 1
        else:
            # Symmetrisch & Power-of-2 nutzen signed range
            q_max = (2 ** (num_bits - 1)) - 1
            q_min = -q_max
            
        # Sicherheitscheck für Scale
        if scale.numel() == 1 and scale.item() == 0:
            scale = torch.tensor(1.0, device=tensor.device)

        return Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
    
