import torch
import copy
import os
import logging
from src.model import CNN
from src.train import train_model
from src.evaluation.evaluate import evaluate
from src.utility.quantization_calibration import calibrate_model
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
    SENSITIVITY_CSV_PATH
)

logger = logging.getLogger("Experiment")

def run_experiment():
    # 1. Daten laden
    logger.info("Lade Datensätze...")
    train_loader, test_loader, num_classes = get_data_loaders()

    # 2. Baseline Model Management
    baseline_path = BASELINE_MODEL_PATH
    model = CNN(num_classes=num_classes).to(DEVICE)

    if os.path.exists(baseline_path):
        logger.info(f"Load Baseline-Model: {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    else:
        logger.info("Kein Baseline-Modell gefunden. Starte Training...")
        model, history = train_model(train_loader, test_loader, num_classes)

    # Sicherstellen, dass Baseline im Float-Modus ist
    if hasattr(model, 'convert_to_baseline'):
        model.convert_to_baseline()
    
    # 3. Baseline Evaluieren
    logger.info("Evaluiere Baseline (Float32)...")
    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32")
    size_base = get_model_size(model)
    logger.info(f"Baseline -> Acc: {acc_base:.2f}%, Time: {time_base:.4f}s, Size: {size_base:.2f}MB")

    plot_training_curves(history)

    # 4. Definition der Experimente
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

    # 5. Systematische Schleife
    for conf in experiment_configs:
        logger.info(f"--- Starte Experiment: {conf['name']} ---")
        
        model_quant = copy.deepcopy(model)

        model_quant.convert_to_quantized(method=conf['method'], bits=conf['bits'])
        
        calibrate_model(model_quant, train_loader, num_batches=10, device=DEVICE)

        acc, time_inf = evaluate(model_quant, test_loader, conf['name'])
        
        drop = acc_base - acc
        
        logger.info(f"Ergebnis {conf['name']}: Acc={acc:.2f}% (Drop: {drop:.2f})")

        logger.info(f"Konvertiere Gewichte zu INT{conf['bits']} für Speicherung...")
        model_quant.permanently_quantize_weights()

        save_path = os.path.join(QUANTIZED_MODELS, f"model_{conf['name']}.pt")
        torch.save(model_quant.state_dict(), save_path)

        real_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"Gespeicherte Modellgröße: {real_size_mb:.2f} MB")
        
        results.append({
            "config_name": conf['name'],
            "method": conf['method'],
            "bits": conf['bits'],
            "accuracy": acc,
            "inference_time": time_inf,
            "model_size_mb": real_size_mb,
            "drop_percentage": drop
        })
        
        save_path = os.path.join(QUANTIZED_MODELS, f"model_{conf['name']}.pt")
        torch.save(model_quant.state_dict(), save_path)

    # 6. Ergebnisse speichern (Funktion kommt jetzt aus utils)
    save_csv(results, EXPERIMENT_CSV_PATH, 
             ['config_name', 'method', 'bits', 'accuracy', 'inference_time', 'model_size_mb', 'drop_percentage'])

    # 7. Sensitivitätsanalyse (Funktion kommt jetzt aus utils)
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