import torch
from torchao.quantization import (
    quantize_,
)
import copy
import os
import logging
from src.model import CNN
from src.train import train_model
from src.evaluation.evaluate import evaluate
from src.utility.utils import (
    get_data_loaders, 
    get_model_size, 
    save_csv, 
    setup_global_logging, 
    run_sensitivity_analysis,
    plot_training_curves
)
from src.utility.config import (
    DEVICE, 
    MODELS_DIR, 
    QUANTIZED_MODELS, 
    BASELINE_MODEL_PATH,
    EXPERIMENT_CONFIGS,
    EXPERIMENT_CSV_PATH,
    SENSITIVITY_CONFIG,   
    SENSITIVITY_CSV_PATH,
    NUM_CLASSES
)

logger = logging.getLogger("Experiment")

def run_experiment():
    # Daten laden
    logger.info("Lade Datensätze...")
    train_loader, test_loader, num_classes = get_data_loaders()

    history = None

    # Baseline Model Management
    baseline_path = BASELINE_MODEL_PATH
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(baseline_path):
        logger.info(f"Load Baseline-Model: {baseline_path}")
        state_dict = torch.load(baseline_path, map_location=DEVICE)
    
        # Extract original tensors
        old_weight = state_dict['fc2.weight'] # Shape: [150, 1024]
        old_bias = state_dict['fc2.bias']     # Shape: [150]
        
        # Create new padded tensors
        new_weight = torch.zeros((152, 1024), device=DEVICE)
        new_bias = torch.zeros((152,), device=DEVICE)
        
        # Copy old data into new tensors
        new_weight[:150, :] = old_weight
        new_bias[:150] = old_bias
        
        # Update the dictionary
        state_dict['fc2.weight'] = new_weight
        state_dict['fc2.bias'] = new_bias
        
        # Load into model (using strict=False to ignore old scale keys)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Kein Baseline-Modell gefunden. Starte Training...")
        model, history = train_model(train_loader, test_loader, num_classes)

    torch.save(model.state_dict(), baseline_path)

    # Sicherstellen, dass Baseline im Float-Modus ist
    if hasattr(model, 'convert_to_baseline'):
        model.convert_to_baseline()
    
    # Baseline Evaluieren
    logger.info("Evaluiere Baseline (Float32)...")
    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32")
    size_base = get_model_size(model)
    logger.info(f"Baseline -> Acc: {acc_base:.2f}%, Time: {time_base:.4f}s, Size: {size_base:.2f}MB")

    if history is not None:
        plot_training_curves(history)
    else:
        logger.info("Baseline loaded. Skipping Training Curve.")

    # Definition der Experimente
    experiment_configs = EXPERIMENT_CONFIGS

    results = []
    
    # Baseline Resultat
    results.append({
        "config_name": "Baseline (Float32)",
        "method": "float",
        "bits": 32,
        "accuracy": acc_base,
        "inference_time": time_base,
        "model_size_mb": size_base,
        "drop_percentage": 0.0
    })

    # Systematische Schleife
    for conf_obj in experiment_configs:
        logger.info(f"--- Starting torchao Experiment: {conf_obj['name']} ---")
        
        model_quant = copy.deepcopy(model)

        quantize_(model_quant, conf_obj['ao_config'])

        model_quant = torch.compile(model_quant, backend="aot_eager")

        acc, time_inf = evaluate(model_quant, test_loader, conf_obj['name'])
        
        drop = acc_base - acc
        
        logger.info(f"Ergebnis {conf_obj['name']}: Acc={acc:.2f}% (Drop: {drop:.2f})")

        save_path = os.path.join(QUANTIZED_MODELS, f"model_{conf_obj['name']}.pt")
        torch.save(model_quant.state_dict(), save_path)

        real_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"Gespeicherte Modellgröße: {real_size_mb:.2f} MB")
        
        results.append({
                "config_name": conf_obj['name'],
                "method": conf_obj['method'],   # Extracted from our hybrid dict
                "bits": conf_obj['bits'],       # Extracted from our hybrid dict
                "accuracy": acc,
                "inference_time": time_inf,
                "model_size_mb": real_size_mb,
                "drop_percentage": drop
            })

    # Ergebnisse speichern (Funktion kommt jetzt aus utils)
    save_csv(results, EXPERIMENT_CSV_PATH, 
             ['config_name', 'method', 'bits', 'accuracy', 'inference_time', 'model_size_mb', 'drop_percentage'])

    # Sensitivitätsanalyse (Funktion kommt jetzt aus utils)
    run_sensitivity_analysis(
        model, 
        test_loader, 
        method=SENSITIVITY_CONFIG['method'], 
        bits=SENSITIVITY_CONFIG['bits']
    )

    logger.info("=== Alle Experimente erfolgreich abgeschlossen ===")

if __name__ == "__main__":
    setup_global_logging()
    run_experiment()