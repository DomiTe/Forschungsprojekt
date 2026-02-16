import torch
import copy
import os
import logging
from src.model import CNN
from src.train import train_model
from src.evaluation.evaluate import evaluate
from src.evaluation.fidelity import calculate_fidelity_metrics
from src.utility.quantization_calibration import calibrate_model
from src.utility.utils import (
    get_data_loaders, 
    get_model_size, 
    save_csv, 
    setup_global_logging, 
    run_sensitivity_analysis,
    plot_training_curves,
    save_quantization_plots  # Ensure this is added to your utils.py
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

    if hasattr(model, 'convert_to_baseline'):
        model.convert_to_baseline()
    
    # 3. Baseline Evaluieren
    logger.info("Evaluiere Baseline (Float32)...")
    base_metrics = evaluate(model, test_loader, "Baseline Float32")
    acc_base = base_metrics['accuracy']
    time_base = base_metrics['inference_time']
    size_base = get_model_size(model)
    logger.info(f"Baseline -> Acc: {acc_base:.2f}%, Time: {time_base:.4f}s, Size: {size_base:.2f}MB")

    # 4. Definition der Experimente
    experiment_configs = EXPERIMENT_CONFIGS
    results = []
    
    # Baseline Resultat
    results.append({
    "config_name": "Baseline (Float32)",
    "method": "float",
    "bits": 32,
    "accuracy": acc_base,
    "precision": base_metrics['precision'],
    "recall": base_metrics['recall'],
    "f1_score": base_metrics['f1_score'],
    "sqnr_db": 100.0,
    "kl_divergence": 0.0,
    "avg_mse": 0.0,
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

        # 1. Deep Fidelity Evaluation (Logits comparison)
        model.eval()
        model_quant.eval()
        total_mse, total_sqnr, total_kl = 0, 0, 0
        num_eval_batches = 10

        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                if i >= num_eval_batches: break 
                imgs = imgs.to(DEVICE)
                b_out, q_out = model(imgs), model_quant(imgs)
                
                mse, sqnr, kl = calculate_fidelity_metrics(b_out, q_out)
                total_mse += mse
                total_sqnr += sqnr
                total_kl += kl

        avg_sqnr = total_sqnr / num_eval_batches
        avg_kl = total_kl / num_eval_batches
        avg_mse = total_mse / num_eval_batches
        
        # 2. Performance Evaluation (Class metrics)
        # Ensure evaluate() returns a DICT with accuracy, precision, recall, f1, inference_time
        eval_res = evaluate(model_quant, test_loader, conf['name'])

        results.append({
            "config_name": conf['name'],
            "method": conf['method'],
            "bits": conf['bits'],
            "accuracy": eval_res['accuracy'],
            "precision": eval_res['precision'],
            "recall": eval_res['recall'],
            "f1_score": eval_res['f1_score'],
            "sqnr_db": avg_sqnr,
            "kl_divergence": avg_kl,
            "avg_mse": avg_mse,
            "inference_time": eval_res['inference_time'],
            "model_size_mb": 0, # Calculated after saving
            "drop_percentage": acc_base - eval_res['accuracy']
        })

        # 3. Save Model State
        model_quant.permanently_quantize_weights()
        save_path = os.path.join(QUANTIZED_MODELS, f"model_{conf['name']}.pt")
        torch.save(model_quant.state_dict(), save_path)
        
        # Update size in the last added result
        results[-1]["model_size_mb"] = os.path.getsize(save_path) / (1024 * 1024)

    # 6. Ergebnisse speichern & Plots generieren
    fieldnames = [
        'config_name', 'method', 'bits', 'accuracy', 'precision', 'recall', 
        'f1_score', 'sqnr_db', 'kl_divergence', 'avg_mse', 'inference_time', 
        'model_size_mb', 'drop_percentage'
    ]

    save_csv(results, EXPERIMENT_CSV_PATH, fieldnames)
    
    # New: Save the visualization plots created in the notebook
    save_quantization_plots(results)

    # 7. Sensitivitätsanalyse
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