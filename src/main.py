import torch
import copy
import os
import logging
import warnings

# Project Imports
from src.model_cnn.model import CNN
from src.model_cnn.train import train_model
from src.evaluation.evaluate import evaluate

from src.torch_quantization.quantization_calibration import calibrate_model
from src.utility.utils import (
    get_data_loaders, 
    get_model_size, 
    save_csv, 
    setup_global_logging, 
    plot_training_curves
)
from src.torch_quantization.quant_utils import (
    fuse_layers,
    get_custome_affine_qconfig,
    get_custome_symmetric_qconfig,
    get_custome_pot_qconfig
)
from src.fake_quantization.fake_quant_config import (
    get_fake_quant_affine_config,
    get_fake_quant_symmetric_config,
    get_fake_quant_pot_config
)
from src.analysis.layer_analysis import LayerAnalyzer
from src.utility.config import (
    DEVICE, 
    QUANTIZED_MODELS, 
    BASELINE_MODEL_PATH,
    EXPERIMENT_CSV_PATH
)

logger = logging.getLogger("Experiment")


def run_experiment():
    warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
    
    # 1. Setup Data
    logger.info("Loading datasets...")
    train_loader, test_loader, num_classes = get_data_loaders()
    torch.backends.quantized.engine = 'x86'

    # 2. Baseline Model
    baseline_path = BASELINE_MODEL_PATH
    model = CNN(num_classes=num_classes).to(DEVICE)

    if os.path.exists(baseline_path):
        logger.info(f"Load Baseline-Model: {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE, weights_only=True), strict=False)
    else:
        logger.info("No baseline found. Starting training...")
        model, history = train_model(train_loader, test_loader, num_classes)
        plot_training_curves(history)

    # 3. Evaluate Baseline
    logger.info("Evaluating Baseline (Float32)...")
    model.eval()
    
    metrics_base = evaluate(model, test_loader, "Baseline Float32", device=torch.device('cpu'))
    acc_base = metrics_base['accuracy']
    size_base = get_model_size(model)

    # Initialize results list with all target columns
    results = [{
        "config_name": "Baseline (Float32)",
        "method": "float",
        "bits": 32,
        "accuracy": acc_base,
        "f1_score": metrics_base['f1_score'],
        "precision": metrics_base['precision'],
        "recall": metrics_base['recall'],
        "inference_time": metrics_base['inference_time'],
        "model_size_mb": size_base,
        "drop_percentage": 0.0
    }]

    # --- Helper for Real PTQ ---
    def run_ptq(exp_name, qconfig_fn, method_key):
        logger.info(f"--- Starting Experiment: {exp_name} ---")
        m = copy.deepcopy(model).to('cpu')
        m.eval()
        m = fuse_layers(m)
        m.qconfig = qconfig_fn()
        torch.ao.quantization.prepare(m, inplace=True)
        calibrate_model(m, train_loader, num_batches=10, device='cpu')
        torch.ao.quantization.convert(m, inplace=True)

        analyzer = LayerAnalyzer(
            model=model, # Your Baseline Float32 model
            loader=test_loader, 
            device='cpu', 
            qconfig=m.qconfig
        )

        logger.info(f"Analyzing real quantization fidelity for {exp_name}...")
        analyzer.run_real_quant_analysis(quantized_model=m, output_csv=f"fidelity_{exp_name}.csv")

        metrics = evaluate(m, test_loader, exp_name, device=torch.device('cpu'))
        save_path = os.path.join(QUANTIZED_MODELS, f"model_{exp_name}.pt")
        torch.jit.save(torch.jit.script(m), save_path)
        
        results.append({
            "config_name": exp_name,
            "method": method_key,
            "bits": 8,
            "accuracy": metrics['accuracy'],
            "f1_score": metrics['f1_score'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "inference_time": metrics['inference_time'],
            "model_size_mb": os.path.getsize(save_path) / (1024 * 1024),
            "drop_percentage": acc_base - metrics['accuracy']
        })
        save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

    # Run Real Quantization
    run_ptq("Affine_PTQ", get_custome_affine_qconfig, "affine")
    run_ptq("Symmetric_PTQ", get_custome_symmetric_qconfig, "symmetric")
    run_ptq("PoT_PTQ", get_custome_pot_qconfig, "pot_manual")

    # ---------------------------------------------------------
    # Experiment 4: Fake Quantization Analysis (Global + Layer-wise)
    # ---------------------------------------------------------
    logger.info("--- Starting Experiment: Detailed Fake Quantization Analysis ---")
    
    model_for_analysis = copy.deepcopy(model).to('cpu')
    model_for_analysis.eval()
    model_for_analysis = fuse_layers(model_for_analysis)
    
    analysis_configs = [
        ("Affine", get_fake_quant_affine_config(), "affine_fake"),
        ("Symmetric", get_fake_quant_symmetric_config(), "symmetric_fake"),
        ("PoT", get_fake_quant_pot_config(), "pot_fake"),
    ]
    
    for method_name, qconfig, method_key in analysis_configs:
        logger.info(f"Evaluating TOTAL performance for: {method_name} (Fake Quant)")
        
        # 1. Create and prepare the simulated model
        model_fake = copy.deepcopy(model_for_analysis)
        model_fake.qconfig = qconfig
        torch.ao.quantization.prepare(model_fake, inplace=True)
        
        # 2. Calibrate the simulated model
        calibrate_model(model_fake, train_loader, num_batches=10, device='cpu')
        
        # 3. Global Evaluation (Total Numbers)
        metrics_fake = evaluate(model_fake, test_loader, f"{method_name} FakeQuant", device=torch.device('cpu'))
        
        # 4. Add to main results CSV for direct comparison
        results.append({
            "config_name": f"{method_name}_FakeSim",
            "method": method_key,
            "bits": 8,
            "accuracy": metrics_fake['accuracy'],
            "f1_score": metrics_fake['f1_score'],
            "precision": metrics_fake['precision'],
            "recall": metrics_fake['recall'],
            "inference_time": metrics_fake['inference_time'],
            "model_size_mb": size_base,
            "drop_percentage": acc_base - metrics_fake['accuracy']
        })
        save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

        # 5. Run Detailed Layer-wise Analyzer (MSE, SQNR, KL)
        analyzer = LayerAnalyzer(
            float_model=model_for_analysis, 
            loader=test_loader, 
            device=torch.device('cpu'),
            qconfig=qconfig
        )
        analyzer.run_layer_wise_analysis(output_csv=f"analysis_fake_quant_{method_name}.csv")

    logger.info("=== All Experiments Completed ===")

if __name__ == "__main__":
    setup_global_logging()
    run_experiment()
