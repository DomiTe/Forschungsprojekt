import torch
import copy
import os
import logging
import warnings
import collections
import statistics

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
    EXPERIMENT_CSV_PATH,
    CSV_DIR
)

logger = logging.getLogger("Experiment")


def run_experiment():
    warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
    
    num_runs = 5
    
    aggregate_results = collections.defaultdict(list)
    aggregate_layer_results = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for run_idx in range(num_runs):
        logger.info(f" Starting Run {run_idx + 1}/{num_runs}")
    # 1. Setup Data
        logger.info("Loading datasets...")
        train_loader, test_loader, num_classes = get_data_loaders()
        torch.backends.quantized.engine = 'x86'

        # 2. Baseline Model
        baseline_path = BASELINE_MODEL_PATH.replace(".pt", f"_run_{run_idx}.pt")
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
        # results = [{
        #     "config_name": "Baseline (Float32)",
        #     "method": "float",
        #     "bits": 32,
        #     "accuracy": acc_base,
        #     "f1_score": metrics_base['f1_score'],
        #     "precision": metrics_base['precision'],
        #     "recall": metrics_base['recall'],
        #     "inference_time": metrics_base['inference_time'],
        #     "model_size_mb": size_base,
        #     "drop_percentage": 0.0
        # }]
        aggregate_results["Baseline (Float32)"].append({
            "method": "float",
            "bits": 32,
            "accuracy": acc_base,
            "f1_score": metrics_base['f1_score'],
            "precision": metrics_base['precision'],
            "recall": metrics_base['recall'],
            "inference_time_ms": metrics_base['inference_time_ms'],
            "model_size_mb": size_base,
            "drop_percentage": 0.0
        })

        # --- Helper for Real PTQ ---
        def run_ptq(exp_name, qconfig_fn, method_key):
            logger.info(f"--- Starting Experiment: {exp_name} ---")
            save_path = os.path.join(QUANTIZED_MODELS, f"model_{exp_name}_run_{run_idx}.pt") 
                       
            # 1. Setup the structure (needed for both loading and training)
            m = copy.deepcopy(model).to('cpu')
            m.eval()
            m = fuse_layers(m)
            current_qconfig = qconfig_fn()
            m.qconfig = current_qconfig
            torch.ao.quantization.prepare(m, inplace=True)

            if os.path.exists(save_path):
                logger.info(f"Quantized state found at {save_path}. Loading...")
                # We must convert to quantized form BEFORE loading the state_dict
                # so the quantized parameters (weight/scale/zp) exist in the object
                torch.ao.quantization.convert(m, inplace=True)
                m.load_state_dict(torch.load(save_path, map_location='cpu'))
            else:
                logger.info("No quantized model found. Running full PTQ pipeline...")
                calibrate_model(m, train_loader, num_batches=10, device='cpu')
                torch.ao.quantization.convert(m, inplace=True)
                # Save state_dict instead of JIT
                torch.save(m.state_dict(), save_path)
            
            # m is now a standard quantized nn.Module, which supports hooks!
            analyzer = LayerAnalyzer(
                float_model=fuse_layers(copy.deepcopy(model).to('cpu')).eval(), 
                loader=test_loader, 
                device='cpu', 
                qconfig=current_qconfig
            )

            logger.info(f"Analyzing real quantization fidelity for {exp_name}...")
            
            layer_csv_name = f"fidelity_{exp_name}_run_{run_idx}.csv"
            layer_metrics = analyzer.run_real_quant_analysis(quantized_model=m, output_csv=layer_csv_name)
            
            for row in layer_metrics:
                layer_name = row["layer_name"]
                aggregate_layer_results[exp_name][layer_name].append(row)
                
            metrics = evaluate(m, test_loader, exp_name, device=torch.device('cpu'))
            
            # results.append({
            #     "config_name": exp_name,
            #     "method": method_key,
            #     "bits": 8,
            #     "accuracy": metrics['accuracy'],
            #     "f1_score": metrics['f1_score'],
            #     "precision": metrics['precision'],
            #     "recall": metrics['recall'],
            #     "inference_time": metrics['inference_time'],
            #     "model_size_mb": os.path.getsize(save_path) / (1024 * 1024),
            #     "drop_percentage": acc_base - metrics['accuracy']
            # })
            
            aggregate_results[exp_name].append({
                "method": method_key,
                "bits": 8,
                "accuracy": metrics['accuracy'],
                "f1_score": metrics['f1_score'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "inference_time_ms": metrics['inference_time_ms'],
                "model_size_mb": os.path.getsize(save_path) / (1024 * 1024),
                "drop_percentage": acc_base - metrics['accuracy']
            })

        # Run Real Quantization
        run_ptq("Affine_PTQ", get_custome_affine_qconfig, "affine")
        run_ptq("Symmetric_PTQ", get_custome_symmetric_qconfig, "symmetric")
        run_ptq("PoT_PTQ", get_custome_pot_qconfig, "pot_manual")
    
    # Calculate averages over all runs
    final_results = []
    
    for config_name, runs in aggregate_results.items():
        # Initialize an average dictionary
        avg_dict = {
            "config_name": config_name,
            "method": runs[0]["method"],
            "bits": runs[0]["bits"],
        }
        
        # Keys to average
        numeric_keys = [
            "accuracy", "f1_score", "precision", "recall", 
            "inference_time_ms", "model_size_mb", "drop_percentage"
        ]
        
        # Calculate sum and divide by number of runs
        for key in numeric_keys:
            values = [run[key] for run in runs]
            
            avg_dict[key] = statistics.mean(values)
            
            avg_dict[f"{key}_std"] = statistics.stdev(values)
            
            
        final_results.append(avg_dict)

    # Save the averaged results
    if final_results:
        save_csv(final_results, EXPERIMENT_CSV_PATH, list(final_results[0].keys()))


    for exp_name, layers in aggregate_layer_results.items():
        final_layer_results = []
        
        for layer_name, runs in layers.items():
            # Initialize the average dictionary for this specific layer
            avg_layer_dict = {
                "layer_name": layer_name
            }
            
            # The keys returned by LayerAnalyzer.run_real_quant_analysis
            layer_numeric_keys = ["avg_mse", "sqnr_db", "kl_divergence"]
            
            # Calculate sum and divide by number of runs using a generator expression
            for key in layer_numeric_keys:
                values = [run[key] for run in runs]
                avg_layer_dict[key] = statistics.mean(values)
                avg_layer_dict[f"{key}_std"] = statistics.stdev(values)
                
            final_layer_results.append(avg_layer_dict)
            
        # Save the averaged layer results for this specific experiment
        if final_layer_results:
            avg_csv_path = os.path.join(CSV_DIR, f"fidelity_{exp_name}_AVG.csv")
            # Note: Ensure CSV_DIR is imported from src.utility.config at the top of main.py
            save_csv(final_layer_results, avg_csv_path, list(final_layer_results[0].keys()))
            
    # ---------------------------------------------------------
    # REDUCTED - Experiment 4: Fake Quantization Analysis - REDUCTED
    # ---------------------------------------------------------
    # logger.info("--- Starting Experiment: Detailed Fake Quantization Analysis ---")
    
    # model_for_analysis = copy.deepcopy(model).to('cpu')
    # model_for_analysis.eval()
    # model_for_analysis = fuse_layers(model_for_analysis)
    
    # analysis_configs = [
    #     ("Affine", get_fake_quant_affine_config(), "affine_fake"),
    #     ("Symmetric", get_fake_quant_symmetric_config(), "symmetric_fake"),
    #     ("PoT", get_fake_quant_pot_config(), "pot_fake"),
    # ]
    
    # for method_name, qconfig, method_key in analysis_configs:
    #     logger.info(f"Evaluating TOTAL performance for: {method_name} (Fake Quant)")
        
    #     # 1. Create and prepare the simulated model
    #     model_fake = copy.deepcopy(model_for_analysis)
    #     model_fake.qconfig = qconfig
    #     torch.ao.quantization.prepare(model_fake, inplace=True)
        
    #     # 2. Calibrate the simulated model
    #     calibrate_model(model_fake, train_loader, num_batches=10, device='cpu')
        
    #     # 3. Global Evaluation (Total Numbers)
    #     metrics_fake = evaluate(model_fake, test_loader, f"{method_name} FakeQuant", device=torch.device('cpu'))
        
    #     # 4. Add to main results CSV for direct comparison
    #     results.append({
    #         "config_name": f"{method_name}_FakeSim",
    #         "method": method_key,
    #         "bits": 8,
    #         "accuracy": metrics_fake['accuracy'],
    #         "f1_score": metrics_fake['f1_score'],
    #         "precision": metrics_fake['precision'],
    #         "recall": metrics_fake['recall'],
    #         "inference_time": metrics_fake['inference_time'],
    #         "model_size_mb": size_base,
    #         "drop_percentage": acc_base - metrics_fake['accuracy']
    #     })
    #     save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

    #     # 5. Run Detailed Layer-wise Analyzer (MSE, SQNR, KL)
    #     analyzer = LayerAnalyzer(
    #         float_model=model_for_analysis, 
    #         loader=test_loader, 
    #         device=torch.device('cpu'),
    #         qconfig=qconfig
    #     )
    #     analyzer.run_layer_wise_analysis(output_csv=f"analysis_fake_quant_{method_name}.csv")

    logger.info("=== All Experiments Completed ===")

if __name__ == "__main__":
    setup_global_logging()
    run_experiment()
