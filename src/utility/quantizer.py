import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Quantization:
    """
    Stellt statische Methoden für verschiedene Quantisierungsstrategien bereit.
    Berechnet Skalierungsfaktoren (Scale) und Nullpunkte (Zero Point).
    """

    @staticmethod
    def _core_quantization(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, q_min: int, q_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Der eigentliche Quantisierungs-Kern (Fake Quantization).
        1. Quantisieren: Float -> Int
        2. Clampen: Wertebereich begrenzen
        3. De-Quantisieren: Int -> Float (Simulation des Fehlers)
        """
        # 1. Quantisieren (Formel: x_int = round(x/S + Z))
        x_int = torch.round((tensor / scale) + zero_point)
        
        # 2. Clampen (Wertebereich sicherstellen)
        x_int = torch.clamp(x_int, q_min, q_max)
        
        # 3. De-Quantisieren (Formel: x_out = (x_int - Z) * S)
        x_dequant = (x_int - zero_point) * scale

        return x_dequant, x_int

    @staticmethod
    def affine_quantization(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Affine Quantisierung (Asymmetrisch).
        Nutzt den vollen Bereich (Min bis Max) der Daten.
        Gut für nicht-zentrierte Daten (z.B. nach ReLU).
        """
        x_min = tensor.min()
        x_max = tensor.max()
        
        # Ziel-Bereich für z.B. uint8: [0, 255]
        q_min = 0
        q_max = (2 ** num_bits) - 1

        # Schutz vor Division durch Null (wenn alle Werte im Tensor gleich sind)
        if x_max == x_min:
            scale = torch.tensor(1.0, device=tensor.device)
            zero_point = torch.tensor(0.0, device=tensor.device)
        else:
            # Formel: S = (xmax - xmin) / (qmax - qmin)
            scale = (x_max - x_min) / float(q_max - q_min)
            # Formel: Z = round(qmin - xmin / S)
            zero_point = torch.round(q_min - (x_min / scale))

        # Werte fake-quantisieren
        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        
        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def symmetric_quantization(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symmetrische Quantisierung.
        Erzwingt, dass der Zero-Point genau 0 ist. 
        Wertebereich ist symmetrisch um 0 (z.B. -127 bis +127).
        """
        # Wir nutzen das betragsmäßige Maximum, um die Skalierung zu bestimmen
        x_abs_max = tensor.abs().max()
        
        # Ziel-Bereich für z.B. int8: [-127, 127]
        # Wir nutzen (2^(N-1) - 1), um -128 zu vermeiden und Symmetrie zu wahren
        q_max = (2 ** (num_bits - 1)) - 1
        q_min = -q_max

        if x_abs_max == 0:
            scale = torch.tensor(1.0, device=tensor.device)
        else:
            scale = x_abs_max / float(q_max)

        zero_point = torch.tensor(0.0, device=tensor.device)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)

        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def power_of_two_quantization(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power-of-2 Quantisierung.
        Ähnlich wie symmetrisch, aber der Scale wird auf die nächste 2er-Potenz gerundet.
        Das erlaubt Bit-Shifts statt Multiplikationen auf Hardware.
        """
        x_abs_max = tensor.abs().max()
        q_max = (2 ** (num_bits - 1)) - 1
        q_min = -q_max

        if x_abs_max == 0:
            scale = torch.tensor(1.0, device=tensor.device)
        else:
            # Berechne den idealen Scale wie bei Symmetrisch
            scale_ideal = x_abs_max / float(q_max)
            
            # Runde Scale auf die nächste 2er Potenz: 2^round(log2(scale))
            scale = 2 ** torch.ceil(torch.log2(scale_ideal))

        zero_point = torch.tensor(0.0, device=tensor.device)

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