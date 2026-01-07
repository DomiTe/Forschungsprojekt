import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm
import glob
# Importiere deine Projekt-Module
from src.model import CNN
from src.utility.utils import get_data_loaders
from src.utility.config import (
    MODEL_SAVE_PATH, 
    QUANTIZED_SAVE_PATH, 
    DEVICE, 
    QUANTIZATION_METHOD, 
    QUANTIZATION_NUM_BITS
)
from src.layers import QuantizedLayers
from src.utility.quantizer import Quantization

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_latest_model(model):
    # Sucht alle .pt Dateien im Models-Ordner
    list_of_files = glob.glob('results/models/*.pt') 
    if not list_of_files:
        logger.error("Keine Modelldatei gefunden! Bitte erst train.py ausführen.")
        return False
        
    # Nimmt die neueste Datei (basierend auf Erstellzeit)
    latest_file = max(list_of_files, key=os.path.getctime)
    logger.info(f"Lade neuestes Modell: {latest_file}")
    
    model.load_state_dict(torch.load(latest_file, map_location=DEVICE))
    return True

def get_activation_stats(model, loader, num_batches):
    """
    Durchläuft einige Batches, um Min/Max der Aktivierungen für jeden Layer zu finden.
    Wir nutzen Hooks, um die Inputs der Layer abzufangen.
    """
    model.eval()
    
    # Dictionary zum Speichern der Min/Max Werte pro Layer
    stats = {}

    def get_hook(layer_name):
        def hook(module, input, output):
            x = input[0].detach()
            
            # Initialisieren oder Updaten der Min/Max Werte
            if layer_name not in stats:
                stats[layer_name] = {"min": x.min(), "max": x.max()}
            else:
                stats[layer_name]["min"] = torch.min(stats[layer_name]["min"], x.min())
                stats[layer_name]["max"] = torch.max(stats[layer_name]["max"], x.max())
        return hook

    # Hooks registrieren
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayers):
            handles.append(module.register_forward_hook(get_hook(name)))

    logger.info(f"Kalibriere Aktivierungen mit {num_batches} Batches...")
    
    # Daten durch das Modell schicken
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(DEVICE)
            model(images)

    # Hooks entfernen
    for h in handles:
        h.remove()
        
    return stats

def apply_calibration(model, stats):
    """
    Berechnet Scale und ZeroPoint basierend auf den gesammelten Stats 
    und schreibt sie in die Layer-Buffer.
    """
    logger.info("Wende Kalibrierungs-Statistiken auf Layer an...")
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayers) and name in stats:
            # Min/Max aus der Kalibrierung holen
            x_min = stats[name]["min"]
            x_max = stats[name]["max"]
            
            # Parameter berechnen
            
            if QUANTIZATION_METHOD == 'symmetric':
                # Symmetrisch: Max(Abs) bestimmt Scale
                abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                # Dummy Tensor für die Berechnung
                _, _, scale, zp = Quantization.symmetric_quantization(
                    torch.tensor([abs_max]).to(DEVICE), 
                    num_bits=QUANTIZATION_NUM_BITS
                )
            else:
                # Affine: Min/Max Differenz bestimmt Scale
                dummy_tensor = torch.tensor([x_min, x_max]).to(DEVICE)
                _, _, scale, zp = Quantization.affine_quantization(
                    dummy_tensor, 
                    num_bits=QUANTIZATION_NUM_BITS
                )
            
            # Werte in den Layer schreiben
            module.act_scale.copy_(scale)
            module.act_zero_point.copy_(zp)
            module.activation_calibrated = True

def main():
    # Daten laden
    train_loader, test_loader, *_ = get_data_loaders()
    
    # FP32 Modell laden
    model = CNN().to(DEVICE)
    if not load_latest_model(model):
        return
    logger.info("FP32 Modell erfolgreich geladen.")

    # Kalibrierung (Statistiken sammeln & anwenden)
    stats = get_activation_stats(model, train_loader, num_batches=200)
    apply_calibration(model, stats)

    # Gewichte Quantisieren (Umwandlung in INT8)
    logger.info(f"Konvertiere Gewichte zu INT{QUANTIZATION_NUM_BITS} ({QUANTIZATION_METHOD})...")
    converted_count = 0
    for module in model.modules():
        if isinstance(module, QuantizedLayers):
            module.quantized_storage(
                num_bits=QUANTIZATION_NUM_BITS, 
                method=QUANTIZATION_METHOD
            )
            converted_count += 1
    logger.info(f"{converted_count} Layer wurden permanent quantisiert.")

    # Validierung
    logger.info("Teste Accuracy des vollständig quantisierten Modells...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testen"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    logger.info(f"Final Accuracy (Quantized): {acc:.2f}%")

    # Speichern
    torch.save(model.state_dict(), QUANTIZED_SAVE_PATH)
    logger.info(f"Quantisiertes Modell gespeichert unter: {QUANTIZED_SAVE_PATH}")

if __name__ == "__main__":
    main()