import torch
import warnings
import copy
from torch.ao.quantization import (
    quantize_fx, 
    QConfigMapping, 
    QConfig, 
    MinMaxObserver, 
    PerChannelMinMaxObserver,
    HistogramObserver,
)

from torch.ao.quantization.backend_config import get_fbgemm_backend_config

from torchao.quantization import (
    quantize_,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    autoquant
)
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
    
    @staticmethod
    def get_custom_qconfig(method="affine"):
        """
        Erstellt eine QConfig basierend auf der gewünschten Methode.
        """
        if method == "affine":
            # Standard x86 Setup: Asymmetrische Aktivierungen, Symmetrische Gewichte
            # Gut für Genauigkeit (nutzt vollen Int8 Bereich für ReLU)
            activation_observer = HistogramObserver.with_args(
                dtype=torch.quint8, 
                qscheme=torch.per_tensor_affine
            )
            weight_observer = PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, 
                qscheme=torch.per_channel_affine
            )

        elif method == "symmetric":
            # Alles Symmetrisch: Zero Point ist immer 0
            # Schneller auf mancher Hardware, aber verliert 1 Bit bei ReLU
            activation_observer = MinMaxObserver.with_args(
                dtype=torch.qint8,  # WICHTIG: qint8 (signed) statt quint8 bei symmetrisch
                qscheme=torch.per_tensor_symmetric
            )
            weight_observer = PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, 
                qscheme=torch.per_channel_symmetric
            )
            
        elif method == "powerof2":
            # Power-of-2 Simulation (PyTorch hat keinen nativen Po2-Only Kernel für CPUs)
            # Wir nutzen Symmetrisch als Basis, da Po2 eine Untermenge davon ist.
            # Für echte Po2 müsste man einen Custom Observer schreiben, der Scales rundet.
            # Für Latenz-Tests ist das hier äquivalent zu "symmetric".
            print("Warnung: Po2 läuft auf CPUs via Standard-Int8 Instruktionen (wie Symmetric).")
            activation_observer = MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
            weight_observer = PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        return QConfig(activation=activation_observer, weight=weight_observer)
    
    @staticmethod
    def apply_fx_quantization(model, data_loader, method, num_batches=30):
        print(f"Starte FX Graph Mode Quantization (Methode: {method})...")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # 1. Vorbereitung
        model_prep = copy.deepcopy(model)
        model_prep.to('cpu')
        model_prep.eval()

        # 2. Config definieren
        if method == "affine":
            # qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
            qconfig = Quantization.get_custom_qconfig(method="affine")
        elif method == "symmetric":
            # qconfig = Quantization.get_custom_qconfig(method="symmetric")
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(
                    dtype=torch.qint8, 
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False
                ),
                weight=PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, 
                    qscheme=torch.per_channel_symmetric
                )
            )
        else:
            raise ValueError(f"Unbekannte FX Methode: {method}")

        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 3. Tracing Input
        example_input = (next(iter(data_loader))[0].to('cpu'), )
        
        # --- FIX: Backend Config Objekt holen ---
        # Strings ('x86') funktionieren in neueren PyTorch Versionen hier nicht mehr
        backend_config = get_fbgemm_backend_config()
        
        # 4. Prepare
        try:
            prepared_model = quantize_fx.prepare_fx(
                model_prep, 
                qconfig_mapping, 
                example_inputs=example_input,
                backend_config=backend_config # <-- Objekt statt String übergeben
            )
        except Exception as e:
            print(f"KRITISCHER FEHLER bei prepare_fx: {e}")
            raise e # Fehler werfen, nicht verschlucken!
        
        # 5. Calibrate
        print(f"Kalibriere mit {num_batches} Batches...")
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_batches: break
                data = data.to('cpu').to(memory_format=torch.channels_last)
                prepared_model(data)
                
        # 6. Convert
        print("Konvertiere Modell (Layer Fusion)...")
        try:
            quantized_model = quantize_fx.convert_fx(
                prepared_model,
                backend_config=backend_config # <-- Auch hier das Objekt nutzen
            )
        except Exception as e:
            print(f"KRITISCHER FEHLER bei convert_fx: {e}")
            raise e

        print("FX Quantisierung abgeschlossen.")
        return quantized_model
    
    @staticmethod
    def apply_torchao_quantization(model, method="dynamic", dummy_input=None):
        print(f"--- TorchAO: Wende '{method}' Quantisierung an ---")
        
        # 1. Methode anwenden
        if method == "dynamic":
            quantize_(model, int8_dynamic_activation_int8_weight())
            
        elif method == "weight_only":
            quantize_(model, int8_weight_only())
            
        elif method == "auto":
            # Hier ist der kritische Teil!
            model = autoquant(model)
            
            if dummy_input is None:
                raise ValueError("Für 'auto' Methode muss ein dummy_input übergeben werden!")
                
            print("Starte AutoQuant Analyse (Eager Mode)...")
            # WICHTIG: Einmal ohne Compile laufen lassen, damit AutoQuant sich entscheidet
            with torch.no_grad():
                model(dummy_input)
            print("AutoQuant hat die beste Strategie gewählt.")
            
        else:
            # Fallback für alte Configs
            if "int4" in method:
                print("Wechsle zu Int8 Weight Only (Int4 braucht GPU-Libs).")
                quantize_(model, int8_weight_only())
            else:
                raise ValueError(f"Unbekannte Methode: {method}")

        # 2. KOMPILIEREN
        print("Kompiliere mit torch.compile (Backend='inductor')...")
        # 'max-autotune' ist auf CachyOS der Turbo
        compiled_model = torch.compile(model, mode="max-autotune", backend="inductor")
        
        return compiled_model
    