import torch
from typing import Tuple
from src.logging import get_logger

logger = get_logger(__name__)


class Quantization:
    @staticmethod
    def affine_quantization(
        tensor: torch.Tensor, num_bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_min = tensor.min()
        x_max = tensor.max()

        q_min = 0
        q_max = 2**num_bits - 1

        if x_max == x_min:
            logger.warning("Tensor min/max are identical.")
            return tensor, tensor.to(torch.uint8)

        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = torch.round(q_min - (x_min / scale))

        logger.debug(
            f"Affine Quantization: Min={x_min:.4f}, Max={x_max:.4f}, "
            f"Scale={scale:.6f}, Zero Point:{zero_point.item()}"
        )

        x_int = torch.round((tensor / scale) + zero_point)
        x_int = torch.clamp(x_int, q_min, q_max)

        x_dequant = (x_int - zero_point) * scale

        return x_dequant, x_int.to(torch.uint8)

    @staticmethod
    def symmetric_quantization(
        tensor: torch.Tensor, num_bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_max = (2 ** (num_bits - 1)) - 1
        q_min = -q_max

        max_val = torch.max(torch.abs(tensor))

        if max_val == 0:
            scale = 1.0
            logger.debug("Symmetric Quantization: Max value is 0, setting scale to 1.0")
        else:
            scale = max_val / q_max

        logger.debug(
            f"Symmetric Quantization: Max value={max_val:.4f}, Scale={scale:.6f}, "
            f"Range=[{q_min}, {q_max}]"
        )

        x_int = torch.round(tensor / scale)
        x_int = torch.clamp(x_int, q_min, q_max)

        x_dequant = x_int * scale

        return x_dequant, x_int.to(torch.int8)
