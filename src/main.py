"""
main.py — train all models on all datasets.

Usage:
    uv run python -m src.main

Trains the full 4-model × 5-dataset matrix and writes a summary CSV to
results/csv/main_summary.csv.

Models:   cnn | resnet18_scratch | resnet18_pretrained | resnet50_pretrained
Datasets: MNIST | FASHION_MNIST | CIFAR10 | CIFAR100 | POKEMON
"""

import os
import sys
import csv
import time
import logging
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.model_cnn.train import train_model, _evaluate, build_model
from src.quantization.train_qat import train_qat
from src.quantization.quantizer import (
    replace_layers_for_quantization,
    calibrate_ptq,
    fuse_model_architectures,
    QuantizedConv2d,
    QuantizedLinear,
)

from src.analysis.hessian import compute_layerwise_hessian_trace
from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.top_eigenvalue import compute_top_eigenvalue
from src.analysis.quant_error import compute_layerwise_quant_error
from src.analysis.classification_metrics import compute_classification_metrics
from src.analysis.benchmark import (
    compare_fp32_vs_int8,
    log_quantization_audit,
)
from src.analysis import validate_pot
from src.analysis import int8_profile
from src.quantization import deploy

from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
)

from src.utility.helper import parse_args
from src.utility.utils import setup_global_logging, get_data_loaders, measure_throughput
from src.utility.config import (
    CSV_DIR,
    LOG_DIR,
    BASE_DIR,
    DATASET_SPECS,
    QUANTIZED_MODELS,
    DEPLOYED_MODELS,
    QAT_EPOCH,
    QAT_LR,
    HESSIAN_BATCH_SIZE,
    RUN_ID,
)



logger = logging.getLogger(__name__)

MODELS = [
    "cnn",
    "resnet18_no_weights",
    "resnet50_no_weights",
]

DATASETS = [
    "CIFAR10",
    # "IMAGENET100",
    
]


def main() -> None:
    args = parse_args()
    local_rank = _setup_distributed()
    total = len(MODELS) * len(DATASETS)
    
    if local_rank == 0:
        setup_global_logging()
        
        logger.info(f"=== Pipeline start: {len(MODELS)} models - {len(DATASETS)} datasets = {total} runs ===")

    # -------------------------------------------------------------------
    # Deploy-Int8-Only mode: reconstruct saved QAT checkpoints, bake PoT
    # weights into standard layers, run the int8 accuracy gate, and exit.
    # Skips FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis.
    # -------------------------------------------------------------------
    if args.deploy_int8_only:
        if local_rank == 0:
            logger.info("=== Deploy-Int8-Only: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        _run_deploy_int8_only(args, local_rank)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Benchmark-Int8-Only mode: reload saved FP32 baselines and deployed
    # full-int8 models, measure GPU latency/throughput/size, and exit.
    # Skips FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis.
    # -------------------------------------------------------------------
    if args.benchmark_int8_only:
        if local_rank == 0:
            logger.info("=== Benchmark-Int8-Only: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        _run_benchmark_int8_only(args, local_rank)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Validate-PoT-Int8 mode: reconstruct the deployed int8 models fresh
    # from saved PTQ/QAT checkpoints and run the per-layer functional
    # PoT-preservation check. Skips FP32/PTQ/QAT training and all
    # Hessian/eigenvalue/SQNR analysis.
    # -------------------------------------------------------------------
    if args.validate_pot_int8:
        if local_rank == 0:
            logger.info("=== Validate-PoT-Int8: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        _run_validate_pot_int8(args, local_rank)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Diagnose-Int8-Perf mode: reconstruct fp32 and deployed int8 models
    # fresh, benchmark int8_weight_only vs int8_dynamic_act vs fp32, and
    # profile CUDA kernel dispatch to diagnose why int8 underperforms fp32.
    # Skips FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis.
    # -------------------------------------------------------------------
    if args.diagnose_int8_perf:
        if local_rank == 0:
            logger.info("=== Diagnose-Int8-Perf: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        _run_diagnose_int8_perf(args, local_rank)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Deploy-CPU-fbgemm mode: convert PoT-quantized PTQ/QAT checkpoints to
    # real INT8 via torch.ao.quantization's fbgemm backend and benchmark
    # accuracy/size/throughput on CPU. Runs as a single local process (no
    # torchrun/distributed init needed) -- see src/quantization/deploy_fbgemm.py.
    # -------------------------------------------------------------------
    if args.deploy_cpu_fbgemm:
        if local_rank == 0:
            logger.info("=== Deploy-CPU-fbgemm: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.quantization.deploy_fbgemm import run_deploy_cpu_fbgemm
        run_deploy_cpu_fbgemm(
            load_run_id=args.load_run_id,
            checkpoint_dir=args.checkpoint_dir,
            eval_subset=args.eval_subset,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Diagnose-Acc-Mismatch mode: isolate why the same checkpoint scores
    # differently locally than on the cluster. Fingerprints the checkpoints,
    # the class-to-index mapping and the transform pipeline, then evaluates
    # the plain FP32 baseline and inspects label/per-class behaviour, and
    # writes results/<RUN_ID>/logs/acc_mismatch_diagnosis.txt for direct
    # diffing against a cluster run of the same mode. All logic lives in
    # src/analysis/diagnose_acc.py -- see its docstring for the hypotheses
    # under test. Runs as a single local process (no torchrun/distributed).
    # -------------------------------------------------------------------
    if args.diagnose_acc_mismatch:
        if local_rank == 0:
            logger.info("=== Diagnose-Acc-Mismatch: skipping training and Hessian/eigenvalue/SQNR analysis ===")
            from src.analysis.diagnose_acc import run_acc_mismatch_diagnosis
            run_acc_mismatch_diagnosis(
                model_name=args.diag_model,
                dataset_name=args.diag_dataset,
                stage=args.diag_stage,
                checkpoint_dir=args.checkpoint_dir,
                load_run_id=args.load_run_id,
                eval_subset=args.eval_subset,
            )
            logger.info("=== Diagnose-Acc-Mismatch complete ===")
        _cleanup()
        return

    summary: list[dict] = []
    ptq_summary: list[dict] = []
    qat_summary: list[dict] = []
    hessian_summary: list[dict] = []
    eigenvalue_summary: list[dict] = []
    quant_error_summary: list[dict] = []
    classification_summary: list[dict] = []
    run_idx = 0
    
    for dataset_name in DATASETS:
        # Load data once per dataset, reuse across all models
        if local_rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading dataset: {dataset_name}")
        try:
            specs = DATASET_SPECS[dataset_name]
            train_loader, val_loader, num_classes = get_data_loaders(dataset_name)
            hessian_loader, _, _ = get_data_loaders(dataset_name, batch_size=HESSIAN_BATCH_SIZE)
        except Exception as exc:
            if local_rank == 0:
                logger.error(f"Failed to load {dataset_name}: {exc}")
            for model_name in MODELS:
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_val_acc": "LOAD_ERROR",
                    "wall_time_min": "",
                    "status": "failed",
                })
            continue

        for model_name in MODELS:
            run_idx += 1
            if local_rank == 0:
                logger.info(f"\n--- Run {run_idx}/{total}: {model_name} on {dataset_name} ---")
            
            t0 = time.perf_counter()
            try:
                # -------------------------------------------------------------
                # FP32 Baseline Training
                # -------------------------------------------------------------
                if args.skip_training:
                    local_rank_device = int(os.environ.get("LOCAL_RANK", 0))
                    device = torch.device(f"cuda:{local_rank_device}" if torch.cuda.is_available() else "cpu")
                    load_run_id = args.load_run_id or RUN_ID
                    model_path = os.path.join(
                        BASE_DIR, "results", load_run_id , "models",
                        f"baseline_{model_name}_{dataset_name}_float32.pt"
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"No saved model at {model_path}"
                        )
                    fp32_model = build_model(
                        num_classes=specs["num_classes"],
                        model_name=model_name,
                        channels=specs["channels"],
                        image_size=specs["image_size"]
                    ).to(device)
                    fp32_model.load_state_dict(
                        torch.load(model_path, map_location=device, weights_only=True)
                    )
                    _, best_val_acc = _evaluate(
                        fp32_model, val_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    history = {"train_acc": [0.0], "val_acc": [best_val_acc]}
                else:
                    fp32_model, history, _ = train_model(
                        train_loader, val_loader, num_classes,
                        model_name=model_name,
                        dataset_name=dataset_name,
                    )
                
                device = next(fp32_model.parameters()).device
                best_val_acc = max(history["val_acc"])
                
                unwrapped_fp32 = fp32_model.module if hasattr(fp32_model, "module") else fp32_model
                
                if local_rank == 0:
                    fp32_class_metrics = compute_classification_metrics(
                        fp32_model, val_loader, device, num_classes=specs["num_classes"]
                    )
                    classification_summary.append({
                        "model": model_name, "dataset": dataset_name, "stage": "FP32",
                        "accuracy": fp32_class_metrics["accuracy"],
                        "precision": fp32_class_metrics["precision"],
                        "recall": fp32_class_metrics["recall"],
                        "f1": fp32_class_metrics["f1"],
                    })
                
                # Measure FP32 Throughput
                dummy_shape = (1, DATASET_SPECS[dataset_name]["channels"], 
                               DATASET_SPECS[dataset_name]["image_size"], 
                               DATASET_SPECS[dataset_name]["image_size"])
                               
                fp32_metrics = measure_throughput(fp32_model, device, dummy_shape)
                
                if local_rank == 0:
                    logger.info("Computing Hessian Trace for FP32...")
                    torch.cuda.empty_cache()
                    with torch.autograd.set_detect_anomaly(True):
                        fp32_traces = compute_layerwise_hessian_trace_pyhessian(
                            unwrapped_fp32, hessian_loader, torch.nn.CrossEntropyLoss(), device
                        )
                    fp32_eigenvalues = compute_top_eigenvalue(
                        unwrapped_fp32, hessian_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    
                    for layer, trace_val in fp32_traces.items():
                        hessian_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "FP32", "layer": layer, "trace": trace_val
                        })
                    for layer, eigenvalue in fp32_eigenvalues.items():
                        eigenvalue_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "FP32", "layer": layer, "eigenvalue": eigenvalue
                        })
                        
                if local_rank == 0:
                    logger.info(f"[FP32] Best Val Acc: {best_val_acc:.2f}% | Latency: {fp32_metrics['latency_ms']:.2f}ms")
                
                fp32_model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
                # -------------------------------------------------------------
                # PTQ Workflow (Power-of-Two + Asymmetric)
                # -------------------------------------------------------------
                if local_rank == 0:
                    logger.info("Starting PTQ Calibration...")
                
                
                
                # Deepcopy to preserve the FP32 weights for QAT later
                ptq_model = copy.deepcopy(unwrapped_fp32)
                del fp32_model
                torch.cuda.empty_cache()
                
                ptq_model.eval()
                
                fuse_model_architectures(ptq_model, model_name)
                # Recursively inject the fake quantizers
                replace_layers_for_quantization(ptq_model)
                ptq_model = ptq_model.to(device)
                
                # Calibrate activation observers using training data
                calibrate_ptq(ptq_model, train_loader, device, num_batches=20)
                
                # Evaluate PTQ accuracy and throughput
                ptq_loss, ptq_acc = _evaluate(ptq_model, val_loader, torch.nn.CrossEntropyLoss(), device)
                
                if local_rank == 0:
                    ptq_class_metrics = compute_classification_metrics(
                        ptq_model, val_loader, device, num_classes=specs["num_classes"]
                    )
                    classification_summary.append({
                        "model": model_name, "dataset": dataset_name, "stage": "PTQ",
                        "accuracy": ptq_class_metrics["accuracy"],
                        "precision": ptq_class_metrics["precision"],
                        "recall": ptq_class_metrics["recall"],
                        "f1": ptq_class_metrics["f1"],
                    })
                if local_rank == 0:
                    ptq_path = os.path.join(QUANTIZED_MODELS, f"ptq_po2_{model_name}_{dataset_name}.pt")
                    torch.save(ptq_model.state_dict(), ptq_path)
                    logger.info(f"Saved uncompiled PTQ state_dict -> {ptq_path}")
                
                if local_rank == 0:
                    logger.info("Computing Hessian Trace for PTQ...")
                    torch.cuda.empty_cache()
                    ptq_traces = compute_layerwise_hessian_trace_pyhessian(
                        ptq_model, hessian_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    ptq_eigenvalues = compute_top_eigenvalue(
                        ptq_model, hessian_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    ptq_quant_error = compute_layerwise_quant_error(
                        unwrapped_fp32, ptq_model 
                    )
                    for layer, trace_val in ptq_traces.items():
                        hessian_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "PTQ", "layer": layer, "trace": trace_val
                        })
                    for layer, eigenvalue in ptq_eigenvalues.items():
                        eigenvalue_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "PTQ", "layer": layer, "eigenvalue": eigenvalue
                        })
                    for layer, metrics in ptq_quant_error.items():
                        quant_error_summary.append({
                            "model":model_name, "dataset": dataset_name,
                            "stage": "PTQ", "layer": layer, "mse" : metrics["mse"], "sqnr": metrics["sqnr"]
                        })
                        
                if local_rank == 0:
                    logger.info("Compiling PTQ model for throughput benchmarking...")
                
                # max-autotune will profile different Triton kernels on your A100 to find the fastest one
                compiled_ptq_model = torch.compile(ptq_model, mode="max-autotune")
                
                ptq_metrics = measure_throughput(compiled_ptq_model, device, dummy_shape)
                
                if local_rank == 0:
                    logger.info(f"[PTQ]  Val Acc: {ptq_acc:.2f}% | Latency: {ptq_metrics['latency_ms']:.2f}ms")
                    ptq_summary.append({
                      "model":        model_name,
                      "dataset":      dataset_name,
                      "fp32_val_acc": f"{best_val_acc:.2f}",
                      "ptq_val_acc":  f"{ptq_acc:.2f}",
                      "acc_drop":     f"{best_val_acc - ptq_acc:.2f}",
                      "fp32_fps":     f"{fp32_metrics['throughput_fps']:.1f}",
                      "ptq_fps":      f"{ptq_metrics['throughput_fps']:.1f}",
                      "speedup":      f"{ptq_metrics['throughput_fps'] / fp32_metrics['throughput_fps']:.2f}",
                      "status":       "ok",
                    })                     
                    # Save the quantized model state
                    ptq_path = os.path.join(QUANTIZED_MODELS, f"ptq_po2_{model_name}_{dataset_name}.pt")
                    torch.save(ptq_model.state_dict(), ptq_path)

                compiled_ptq_model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
                # -------------------------------------------------------------
                # QAT Workflow (Building on PTQ)
                # -------------------------------------------------------------
                if local_rank == 0:
                    logger.info("Starting Quantization-Aware Training (QAT)...")
                
                # Note: We pass the uncompiled ptq_model directly to QAT
                qat_model, qat_history, _ = train_qat(
                    ptq_model=ptq_model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    device=device,
                    epochs=QAT_EPOCH,
                    lr=QAT_LR
                )
                
                qat_acc = max(qat_history["val_acc"])

                if local_rank == 0:
                    qat_class_metrics = compute_classification_metrics(
                        qat_model, val_loader, device, num_classes=specs["num_classes"]
                    )
                    classification_summary.append({
                        "model": model_name, "dataset": dataset_name, "stage": "QAT",
                        "accuracy": qat_class_metrics["accuracy"],
                        "precision": qat_class_metrics["precision"],
                        "recall": qat_class_metrics["recall"],
                        "f1": qat_class_metrics["f1"],
                    })
                
                if local_rank == 0:
                    logger.info("Computing Hessian Trace for QAT...")
                    torch.cuda.empty_cache()
                    qat_traces = compute_layerwise_hessian_trace_pyhessian(
                        qat_model, hessian_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    
                    qat_eigenvalues = compute_top_eigenvalue(
                        qat_model, hessian_loader, torch.nn.CrossEntropyLoss(), device
                    )
                    qat_quant_error = compute_layerwise_quant_error(
                        unwrapped_fp32, qat_model
                    )
                    for layer, trace_val in qat_traces.items():
                        hessian_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "QAT", "layer": layer, "trace": trace_val
                        })
                    for layer, eigenvalue in qat_eigenvalues.items():
                        eigenvalue_summary.append({
                            "model": model_name, "dataset": dataset_name,
                            "stage": "QAT", "layer": layer, "eigenvalue": eigenvalue
                        })
                    for layer, metrics in qat_quant_error.items():
                        quant_error_summary.append({
                            "model":model_name, "dataset": dataset_name,
                            "stage": "QAT", "layer": layer, "mse" : metrics["mse"], "sqnr": metrics["sqnr"]
                        })
                        
                
                if local_rank == 0:
                    qat_path = os.path.join(QUANTIZED_MODELS, f"qat_po2_{model_name}_{dataset_name}.pt")
                    torch.save(qat_model.state_dict(), qat_path)

                compiled_qat_model = torch.compile(qat_model, mode="max-autotune")
                qat_metrics = measure_throughput(compiled_qat_model, device, dummy_shape)
                
                if local_rank == 0:
                    logger.info(f"[QAT]  Val Acc: {qat_acc:.2f}% | Latency: {qat_metrics['latency_ms']:.2f}ms")
                    qat_summary.append({
                        "model":        model_name,
                        "dataset":      dataset_name,
                        "ptq_val_acc":  f"{ptq_acc:.2f}",
                        "qat_val_acc":  f"{qat_acc:.2f}",
                        "acc_recovered":f"{qat_acc - ptq_acc:.2f}",
                        "fp32_fps":     f"{fp32_metrics['throughput_fps']:.1f}",
                        "qat_fps":      f"{qat_metrics['throughput_fps']:.1f}",
                        "speedup":      f"{qat_metrics['throughput_fps'] / fp32_metrics['throughput_fps']:.2f}",
                        "status":       "ok",
                    })
                    
                # if local_rank == 0:
                #     logger.info("Converting float32 qat to int8")
                # int8_qat_model = copy.deepcopy(qat_model).eval()
                # from src.quantization.convert import convert_qat_to_real
                # convert_qat_to_real(int8_qat_model)
                # int8_qat_model = int8_qat_model.to(device)
                
                # compiled_int8_qat_model = torch.compile(int8_qat_model, mode="max-autotune")
                # int8_metrics = measure_throughput(compiled_int8_qat_model, device, dummy_shape)
                
                # if local_rank == 0:
                #     logger.info(f"[True Int8] Latency: {int8_metrics['latency_ms']: .3f} | FPS: {int8_metrics['throughput_fps']: .3f}")
                    
                # -------------------------------------------------------------
                # Summary Logging Updates
                # -------------------------------------------------------------
                elapsed_min = (time.perf_counter() - t0) / 60
                
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_train_acc": f"{max(history['train_acc']):.2f}",
                    "best_val_acc": f"{best_val_acc:.2f}",
                    "ptq_val_acc": f"{ptq_acc:.2f}",
                    "qat_val_acc": f"{qat_acc:.2f}",  # New QAT Accuracy
                    "fp32_fps": f"{fp32_metrics['throughput_fps']:.1f}",
                    "ptq_fps": f"{ptq_metrics['throughput_fps']:.1f}",
                    "qat_fps": f"{qat_metrics['throughput_fps']:.1f}", # New QAT FPS
                    "wall_time_min": f"{elapsed_min:.1f}",
                    "status": "ok",
                })
                
                compiled_qat_model.zero_grad(set_to_none=True)
                del compiled_ptq_model, compiled_qat_model, ptq_model, qat_model
                torch.cuda.empty_cache()
                
                # compiled_int8_qat_model.zero_grad(set_to_none=True)
                # del int8_qat_model, compiled_int8_qat_model
                # torch.cude.empty_cache()
                
            except Exception as exc:
                elapsed_min = (time.perf_counter() - t0) / 60
                if local_rank == 0:
                    logger.error(f"FAILED {model_name}/{dataset_name}: {exc}", exc_info=True)
                
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_train_acc": "ERROR",
                    "best_val_acc": "ERROR",
                    "ptq_val_acc": "ERROR",
                    "qat_val_acc": "ERROR",
                    "fp32_fps": "ERROR",
                    "ptq_fps": "ERROR",
                    "qat_fps": "ERROR",
                    "wall_time_min": f"{elapsed_min:.1f}",
                    "status": "failed",
                })
                ptq_summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "fp32_val_acc": "ERROR", "ptq_val_acc": "ERROR", "acc_drop": "ERROR",
                    "fp32_fps": "ERROR", "ptq_fps": "ERROR", "speedup": "ERROR",
                    "status": "failed",
                })
                qat_summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "ptq_val_acc": "ERROR", "qat_val_acc": "ERROR", "acc_recovered": "ERROR",
                    "fp32_fps": "ERROR", "qat_fps": "ERROR", "speedup": "ERROR",
                    "status": "failed",
                })
            finally:
                import gc
                for name in ['fp32_model', 'ptq_model', 'qat_model',
                            'compiled_ptq_model', 'compiled_qat_model']:
                    if name in dir():
                        exec(f'del {name}')
                gc.collect()
                torch.cuda.empty_cache()
                
    if local_rank == 0:
        _save_csv_summary(summary, [
            "model", "dataset", "best_train_acc", "best_val_acc",
            "ptq_val_acc", "qat_val_acc", "fp32_fps", "ptq_fps", "qat_fps",
            "wall_time_min", "status",
        ], "pipeline_summary.csv")

        _save_csv_summary(ptq_summary, [
            "model", "dataset", "fp32_val_acc", "ptq_val_acc", "acc_drop",
            "fp32_fps", "ptq_fps", "speedup", "status",
        ], "ptq_summary.csv")

        _save_csv_summary(qat_summary, [
            "model", "dataset", "ptq_val_acc", "qat_val_acc", "acc_recovered",
            "fp32_fps", "qat_fps", "speedup", "status",
        ], "qat_summary.csv")
        
        if hessian_summary:
            df = pd.DataFrame(hessian_summary)
            df.to_csv(os.path.join(CSV_DIR, "layerwise_hessian_traces.csv"), index=False)
            logger.info(f"Hessian Traces saved -> layerwise_hessian_traces.csv")
        if eigenvalue_summary:
            df = pd.DataFrame(eigenvalue_summary)
            df.to_csv(os.path.join(CSV_DIR, "layerwise_top_eigenvalues.csv"), index=False)
            logger.info("Top Eigenvalues saved -> layerwise_top_eigenvalues.csv")
        if quant_error_summary:
            df = pd.DataFrame(quant_error_summary)
            df.to_csv(os.path.join(CSV_DIR, "layerwise_quant_error.csv"), index=False)
            logger.info("MSE and SQNR log saved -> layerwise_quant_error.csv")
        if classification_summary:
            df = pd.DataFrame(classification_summary)
            df.to_csv(os.path.join(CSV_DIR, "classification_metrics.csv") , index=False)
            logger.info("Classification saved -> classification_metrics.csv")
            
        _print_summary(summary)
        logger.info("=== Pipeline complete ===")
    _cleanup()

# Deploy-Int8-Only is restricted to this subset of DATASETS regardless of
# what the training path has enabled above.
DEPLOY_DATASETS = ["CIFAR10", "IMAGENET100"]

# (stage label, checkpoint filename prefix) — both PTQ and QAT checkpoints
# went through fuse_model_architectures + replace_layers_for_quantization,
# so they share the same custom-quantized-layer structure and loader.
DEPLOY_STAGES = [
    ("PTQ", "ptq_po2"),
    ("QAT", "qat_po2"),
]


def _run_deploy_int8_only(args, local_rank: int) -> None:
    load_run_id = args.load_run_id or RUN_ID
    local_rank_device = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank_device}" if torch.cuda.is_available() else "cpu")

    deployment_summary: list[dict] = []

    for dataset_name in DEPLOY_DATASETS:
        if local_rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"[Deploy-Int8] Loading dataset: {dataset_name}")
        try:
            specs = DATASET_SPECS[dataset_name]
            _, val_loader, _ = get_data_loaders(dataset_name)
        except Exception as exc:
            if local_rank == 0:
                logger.error(f"Failed to load {dataset_name}: {exc}")
                for model_name in MODELS:
                    for stage_name, _ in DEPLOY_STAGES:
                        deployment_summary.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            "pot_baked_acc": "LOAD_ERROR",
                            "weight_only_int8_acc": "LOAD_ERROR",
                            "full_int8_acc": "LOAD_ERROR",
                        })
            continue

        for model_name in MODELS:
            for stage_name, checkpoint_prefix in DEPLOY_STAGES:
                if local_rank == 0:
                    logger.info(f"\n--- Deploy-Int8 {stage_name} {model_name} on {dataset_name} ---")
                try:
                    # ---------------------------------------------------------
                    # Reconstruct and load the saved PTQ/QAT model
                    # ---------------------------------------------------------
                    checkpoint_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "quantized_models",
                        f"{checkpoint_prefix}_{model_name}_{dataset_name}.pt"
                    )
                    if not os.path.exists(checkpoint_path):
                        raise FileNotFoundError(f"No saved {stage_name} model at {checkpoint_path}")

                    # ---------------------------------------------------------
                    # Reconstruction + quantization both go through the
                    # shared builder now (src.quantization.deploy), which
                    # asserts Conv2d AND Linear actually got quantized. See
                    # deploy.py's docstring: a bare quantize_(model,
                    # Int8DynamicActivationInt8WeightConfig()) here used to
                    # silently skip every Conv2d layer.
                    # ---------------------------------------------------------
                    pot_baked_model, full_int8_model, _ = deploy.build_int8_model(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        stage=stage_name,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        num_classes=specs["num_classes"],
                        channels=specs["channels"],
                        image_size=specs["image_size"],
                    )
                    pot_baked_acc = evaluate(pot_baked_model, val_loader, device)
                    full_int8_acc = evaluate(full_int8_model, val_loader, device)

                    # ---------------------------------------------------------
                    # Weight-only int8 — PoT-preservation gate. Deliberately
                    # a different, Linear-only config (Int8WeightOnlyConfig's
                    # default filter_fn leaves Conv2d untouched) -- this is
                    # an intentional diagnostic comparison, not "the"
                    # deployed model, so it stays outside the shared builder.
                    # ---------------------------------------------------------
                    weight_only_model = copy.deepcopy(pot_baked_model)
                    quantize_(weight_only_model, Int8WeightOnlyConfig())
                    weight_only_int8_acc = evaluate(weight_only_model, val_loader, device)

                    if local_rank == 0:
                        logger.info(
                            f"[Deploy-Int8] {stage_name} {model_name}/{dataset_name} | "
                            f"PoT baked: {pot_baked_acc:.2f}% | "
                            f"Weight-only int8: {weight_only_int8_acc:.2f}% | "
                            f"Full int8: {full_int8_acc:.2f}%"
                        )

                        # -----------------------------------------------------
                        # Save the deployed int8 models. These state dicts
                        # contain torchao tensor subclasses, so reloading them
                        # elsewhere requires `import torchao` before torch.load.
                        # -----------------------------------------------------
                        weight_only_path = os.path.join(
                            DEPLOYED_MODELS,
                            f"deployed_weightonly_{stage_name}_{model_name}_{dataset_name}.pt"
                        )
                        full_int8_path = os.path.join(
                            DEPLOYED_MODELS,
                            f"deployed_full_{stage_name}_{model_name}_{dataset_name}.pt"
                        )
                        torch.save(weight_only_model.state_dict(), weight_only_path)
                        torch.save(full_int8_model.state_dict(), full_int8_path)
                        logger.info(f"Saved deployed int8 models -> {weight_only_path}, {full_int8_path}")

                        # -----------------------------------------------------
                        # Round-trip check on the full-int8 model: reload the
                        # saved state dict into a freshly-baked+quantized
                        # skeleton and confirm accuracy matches, to catch
                        # torchao serialization issues early.
                        # -----------------------------------------------------
                        _, reload_model, _ = deploy.build_int8_model(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            stage=stage_name,
                            checkpoint_path=checkpoint_path,
                            device=device,
                            num_classes=specs["num_classes"],
                            channels=specs["channels"],
                            image_size=specs["image_size"],
                        )
                        # torchao tensor subclasses require `import torchao`
                        # (done at module import time above) before this load.
                        reload_model.load_state_dict(
                            torch.load(full_int8_path, map_location=device, weights_only=False)
                        )
                        reload_model.eval()
                        reloaded_full_int8_acc = evaluate(reload_model, val_loader, device)

                        if abs(reloaded_full_int8_acc - full_int8_acc) > 0.01:
                            logger.warning(
                                f"[Deploy-Int8] Round-trip mismatch for {stage_name} {model_name}/{dataset_name}: "
                                f"pre-save full int8 acc {full_int8_acc:.4f}% vs reloaded {reloaded_full_int8_acc:.4f}%"
                            )

                        deployment_summary.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            "pot_baked_acc": f"{pot_baked_acc:.2f}",
                            "weight_only_int8_acc": f"{weight_only_int8_acc:.2f}",
                            "full_int8_acc": f"{full_int8_acc:.2f}",
                        })

                        del reload_model

                    del pot_baked_model, weight_only_model, full_int8_model

                except Exception as exc:
                    if local_rank == 0:
                        logger.error(f"FAILED {stage_name} {model_name}/{dataset_name}: {exc}", exc_info=True)
                        deployment_summary.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            "pot_baked_acc": "ERROR",
                            "weight_only_int8_acc": "ERROR",
                            "full_int8_acc": "ERROR",
                        })
                finally:
                    torch.cuda.empty_cache()

    if local_rank == 0:
        _save_csv_summary(deployment_summary, [
            "model", "dataset", "stage", "pot_baked_acc", "weight_only_int8_acc", "full_int8_acc",
        ], "deployment_int8.csv")
        logger.info("=== Deploy-Int8-Only complete ===")


# Benchmark-Int8-Only reuses the same dataset/stage scope as Deploy-Int8-Only.
BENCHMARK_DATASETS = DEPLOY_DATASETS
BENCHMARK_STAGES = DEPLOY_STAGES


def _run_benchmark_int8_only(args, local_rank: int) -> None:
    load_run_id = args.load_run_id or RUN_ID
    local_rank_device = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank_device}" if torch.cuda.is_available() else "cpu")

    sweep_summary: list[dict] = []
    fit_summary: list[dict] = []

    for dataset_name in BENCHMARK_DATASETS:
        specs = DATASET_SPECS[dataset_name]
        base_input_shape = (specs["channels"], specs["image_size"], specs["image_size"])

        for model_name in MODELS:
            for stage_name, checkpoint_prefix in BENCHMARK_STAGES:
                if local_rank == 0:
                    logger.info(f"\n--- Benchmark-Int8 {stage_name} {model_name} on {dataset_name} ---")
                try:
                    # ---------------------------------------------------------
                    # Reload the FP32 baseline
                    # ---------------------------------------------------------
                    fp32_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "models",
                        f"baseline_{model_name}_{dataset_name}_float32.pt"
                    )
                    if not os.path.exists(fp32_path):
                        raise FileNotFoundError(f"No saved FP32 baseline at {fp32_path}")

                    fp32_model = build_model(
                        num_classes=specs["num_classes"],
                        model_name=model_name,
                        channels=specs["channels"],
                        image_size=specs["image_size"]
                    ).to(device)
                    fp32_model.load_state_dict(
                        torch.load(fp32_path, map_location=device, weights_only=True)
                    )
                    fp32_model.eval()

                    # ---------------------------------------------------------
                    # Reconstruct the int8 model FRESH — the same path
                    # Deploy-Int8-Only proved preserves accuracy: rebuild the
                    # quantized-layer skeleton, load the PTQ/QAT checkpoint,
                    # bake PoT weights into standard layers, then apply the
                    # full-int8 torchao config.
                    #
                    # We deliberately do NOT save/reload this model. torchao
                    # quantized state dicts do not reliably round-trip through
                    # load_state_dict into a fresh skeleton — an earlier
                    # version of this code did that and silently ended up
                    # benchmarking an fp32 model under the "int8" label.
                    # ---------------------------------------------------------
                    checkpoint_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "quantized_models",
                        f"{checkpoint_prefix}_{model_name}_{dataset_name}.pt"
                    )
                    if not os.path.exists(checkpoint_path):
                        raise FileNotFoundError(f"No saved {stage_name} model at {checkpoint_path}")

                    # Reconstruction + quantization both go through the
                    # shared builder now (src.quantization.deploy), which
                    # applies dynamic-activation int8 to Linear and
                    # weight-only int8 to Conv2d (see apply_int8_quantization's
                    # docstring in benchmark.py) and asserts both actually
                    # got quantized -- fails loudly rather than silently
                    # benchmarking fp32 numbers under an "int8" label.
                    label = f"{stage_name} {model_name}/{dataset_name}"
                    _, int8_model, audit_details = deploy.build_int8_model(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        stage=stage_name,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        num_classes=specs["num_classes"],
                        channels=specs["channels"],
                        image_size=specs["image_size"],
                    )

                    quant_summary = None
                    if local_rank == 0:
                        quant_summary = log_quantization_audit(audit_details, label)

                    # ---------------------------------------------------------
                    # FP32 vs int8 latency/throughput/size sweep. Both models
                    # are benchmarked uncompiled, so the comparison isolates
                    # precision rather than compile effects. Benchmarking
                    # torch.compile'd variants could be a useful follow-up,
                    # but both sides would need to be compiled for a fair
                    # comparison.
                    # ---------------------------------------------------------
                    with torch.no_grad():
                        result = compare_fp32_vs_int8(
                            fp32_model, int8_model, base_input_shape, device
                        )

                    if local_rank == 0:
                        logger.info(
                            f"[Benchmark-Int8] {stage_name} {model_name}/{dataset_name} | "
                            f"size {result['fp32_size_bytes'] / 1024**2:.2f}MB -> "
                            f"{result['int8_size_bytes'] / 1024**2:.2f}MB "
                            f"({result['size_reduction_x']:.2f}x) | "
                            f"compute speedup {result['compute_speedup_x']:.2f}x"
                        )

                        for row in result["sweep"]:
                            sweep_summary.append({
                                "model": model_name,
                                "dataset": dataset_name,
                                "stage": stage_name,
                                "batch": row["batch"],
                                "fp32_latency_ms": f"{row['fp32_latency_ms']:.4f}",
                                "int8_latency_ms": f"{row['int8_latency_ms']:.4f}",
                                "fp32_throughput_ips": f"{row['fp32_throughput_ips']:.1f}",
                                "int8_throughput_ips": f"{row['int8_throughput_ips']:.1f}",
                                "speedup_x": f"{row['speedup_x']:.2f}",
                            })

                        fp32_fit = result["fp32_fit"]
                        int8_fit = result["int8_fit"]
                        fit_summary.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            "fp32_size_mb": f"{result['fp32_size_bytes'] / 1024**2:.2f}",
                            "int8_size_mb": f"{result['int8_size_bytes'] / 1024**2:.2f}",
                            "size_reduction_x": f"{result['size_reduction_x']:.2f}",
                            "fp32_intercept_ms": f"{fp32_fit['intercept_ms']:.4f}",
                            "int8_intercept_ms": f"{int8_fit['intercept_ms']:.4f}",
                            "fp32_slope_ms": f"{fp32_fit['slope_ms_per_sample']:.4f}",
                            "int8_slope_ms": f"{int8_fit['slope_ms_per_sample']:.4f}",
                            "fp32_r2": f"{fp32_fit['r2']:.4f}",
                            "int8_r2": f"{int8_fit['r2']:.4f}",
                            "compute_speedup_x": f"{result['compute_speedup_x']:.2f}",
                            "decomposition_reliable": result["decomposition_reliable"],
                            "conv_quantized": quant_summary["conv_quantized"],
                        })

                    del fp32_model, int8_model

                except Exception as exc:
                    if local_rank == 0:
                        logger.error(f"FAILED {stage_name} {model_name}/{dataset_name}: {exc}", exc_info=True)
                        fit_summary.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            "fp32_size_mb": "ERROR",
                            "int8_size_mb": "ERROR",
                            "size_reduction_x": "ERROR",
                            "fp32_intercept_ms": "ERROR",
                            "int8_intercept_ms": "ERROR",
                            "fp32_slope_ms": "ERROR",
                            "int8_slope_ms": "ERROR",
                            "fp32_r2": "ERROR",
                            "int8_r2": "ERROR",
                            "compute_speedup_x": "ERROR",
                            "decomposition_reliable": "ERROR",
                            "conv_quantized": "ERROR",
                        })
                finally:
                    torch.cuda.empty_cache()

    if local_rank == 0:
        _save_csv_summary(sweep_summary, [
            "model", "dataset", "stage", "batch",
            "fp32_latency_ms", "int8_latency_ms",
            "fp32_throughput_ips", "int8_throughput_ips", "speedup_x",
        ], "benchmark_sweep.csv")

        _save_csv_summary(fit_summary, [
            "model", "dataset", "stage", "fp32_size_mb", "int8_size_mb", "size_reduction_x",
            "fp32_intercept_ms", "int8_intercept_ms", "fp32_slope_ms", "int8_slope_ms",
            "fp32_r2", "int8_r2", "compute_speedup_x", "decomposition_reliable", "conv_quantized",
        ], "benchmark_summary.csv")
        logger.info("=== Benchmark-Int8-Only complete ===")


# Validate-PoT-Int8 reuses the same dataset/stage scope as Deploy/Benchmark-Int8-Only.
VALIDATE_POT_DATASETS = DEPLOY_DATASETS
VALIDATE_POT_STAGES = DEPLOY_STAGES


def _run_validate_pot_int8(args, local_rank: int) -> None:
    """
    Orchestrates validate_pot's per-layer PoT-preservation check across the
    model/dataset/stage matrix. All actual reconstruction/comparison logic
    lives in src.analysis.validate_pot; this just loops, and rank-0-guards
    logging/CSV/plotting.

    Unlike Deploy/Benchmark-Int8-Only, this needs no data loader (inputs are
    synthetic, fixed-seed) and no DDP-wrapped forward pass, so there is
    nothing for non-zero ranks to usefully do -- the whole per-run body is
    guarded, not just the logging tail. dist.destroy_process_group() in
    _cleanup() is still called by every rank afterward.
    """
    load_run_id = args.load_run_id or RUN_ID
    local_rank_device = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank_device}" if torch.cuda.is_available() else "cpu")

    if local_rank != 0:
        return

    csv_rows: list[dict] = []
    plotted_models: set[str] = set()

    for dataset_name in VALIDATE_POT_DATASETS:
        specs = DATASET_SPECS[dataset_name]

        for model_name in MODELS:
            for stage_name, checkpoint_prefix in VALIDATE_POT_STAGES:
                logger.info(f"\n--- Validate-PoT-Int8 {stage_name} {model_name} on {dataset_name} ---")
                try:
                    checkpoint_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "quantized_models",
                        f"{checkpoint_prefix}_{model_name}_{dataset_name}.pt"
                    )
                    baked_model, int8_model, _ = deploy.build_int8_model(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        stage=stage_name,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        num_classes=specs["num_classes"],
                        channels=specs["channels"],
                        image_size=specs["image_size"],
                    )

                    layer_results = validate_pot.compare_pot_vs_int8_layers(
                        baked_model, int8_model, device
                    )
                    validate_pot.log_and_summarize_pot_validation(
                        layer_results, model_name, dataset_name, stage_name
                    )
                    csv_rows.extend(
                        validate_pot.build_csv_rows(layer_results, model_name, dataset_name, stage_name)
                    )

                    if model_name not in plotted_models:
                        try:
                            validate_pot.plot_pot_weight_histogram(
                                baked_model, model_name, dataset_name, stage_name, LOG_DIR
                            )
                            plotted_models.add(model_name)
                        except Exception as exc:
                            logger.warning(
                                f"[ValidatePoT] Failed to save PoT weight histogram for {model_name}: {exc}"
                            )

                    del baked_model, int8_model

                except Exception as exc:
                    logger.error(f"FAILED {stage_name} {model_name}/{dataset_name}: {exc}", exc_info=True)
                finally:
                    torch.cuda.empty_cache()

    _save_csv_summary(csv_rows, validate_pot.CSV_FIELDNAMES, "pot_validation.csv")
    logger.info("=== Validate-PoT-Int8 complete ===")


# Diagnose-Int8-Perf reuses the same dataset/stage scope and batch sizes as
# the rest of the *_int8_only pipeline modes.
DIAGNOSE_INT8_PERF_DATASETS = DEPLOY_DATASETS
DIAGNOSE_INT8_PERF_STAGES = DEPLOY_STAGES


def _run_diagnose_int8_perf(args, local_rank: int) -> None:
    """
    Orchestrates int8_profile's fp32-vs-int8 performance diagnosis across
    the model/dataset/stage matrix. All reconstruction/benchmarking/
    profiling logic lives in src.analysis.int8_profile; this just loops,
    collects results, and writes the sweep CSV + combined text report.

    Like Validate-PoT-Int8, this needs no data loader and no DDP-wrapped
    forward pass (synthetic fixed-shape inputs only), so the whole body is
    rank-0-guarded rather than duplicating identical benchmarking/profiling
    work across ranks.
    """
    load_run_id = args.load_run_id or RUN_ID
    local_rank_device = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank_device}" if torch.cuda.is_available() else "cpu")

    if local_rank != 0:
        return

    gpu_info = int8_profile.get_gpu_capability_info(device)

    sweep_csv_rows: list[dict] = []
    report_sections: list[str] = []

    for dataset_name in DIAGNOSE_INT8_PERF_DATASETS:
        specs = DATASET_SPECS[dataset_name]

        for model_name in MODELS:
            for stage_name, checkpoint_prefix in DIAGNOSE_INT8_PERF_STAGES:
                logger.info(f"\n--- Diagnose-Int8-Perf {stage_name} {model_name} on {dataset_name} ---")
                try:
                    fp32_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "models",
                        f"baseline_{model_name}_{dataset_name}_float32.pt"
                    )
                    checkpoint_path = os.path.join(
                        BASE_DIR, "results", load_run_id, "quantized_models",
                        f"{checkpoint_prefix}_{model_name}_{dataset_name}.pt"
                    )

                    result = int8_profile.run_int8_perf_diagnosis(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        stage=stage_name,
                        channels=specs["channels"],
                        num_classes=specs["num_classes"],
                        image_size=specs["image_size"],
                        fp32_checkpoint_path=fp32_path,
                        checkpoint_path=checkpoint_path,
                        device=device,
                    )

                    for row in result["sweep_rows"]:
                        sweep_csv_rows.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "stage": stage_name,
                            **row,
                        })

                    report_sections.append(
                        int8_profile.build_int8_perf_report(
                            model_name, dataset_name, stage_name,
                            result["sweep_rows"], result["fp32_profile"], result["int8_profile"],
                            result["conv_quantized"], result["linear_quantized"],
                        )
                    )

                except Exception as exc:
                    logger.error(f"FAILED {stage_name} {model_name}/{dataset_name}: {exc}", exc_info=True)
                finally:
                    torch.cuda.empty_cache()

    _save_csv_summary(sweep_csv_rows, int8_profile.SWEEP_CSV_FIELDNAMES, "int8_perf_diagnosis_sweep.csv")

    report_path = os.path.join(LOG_DIR, "int8_perf_diagnosis.txt")
    int8_profile.write_report(report_path, gpu_info, report_sections)
    logger.info("=== Diagnose-Int8-Perf complete ===")


def bake_pot_into_standard_layers(model: nn.Module) -> nn.Module:
    # Operates on a deepcopy, so the original quantized model is left untouched.
    baked_model = copy.deepcopy(model)

    def _bake(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, QuantizedConv2d):
                plain = nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None)
                ).to(child.weight.device)

                plain.weight.data.copy_(child.weight_fake_quant(child.weight).data)
                if child.bias is not None:
                    plain.bias.data.copy_(child.bias.data)

                setattr(module, name, plain)

            elif isinstance(child, QuantizedLinear):
                plain = nn.Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None)
                ).to(child.weight.device)

                plain.weight.data.copy_(child.weight_fake_quant(child.weight).data)
                if child.bias is not None:
                    plain.bias.data.copy_(child.bias.data)

                setattr(module, name, plain)

            else:
                _bake(child)

    _bake(baked_model)
    return baked_model


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    # Returns top-1 accuracy as a percentage (0-100), not a 0-1 fraction.
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def _save_csv_summary(rows: list[dict], fieldnames: list[str], filename: str) -> None:
    path = os.path.join(CSV_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary saved → {path}")


def _print_summary(summary: list[dict]) -> None:
    logger.info("\n=== PIPELINE SUMMARY ===")
    header = f"{'Model':<24} {'Dataset':<16}  {'Best Train Acc':>12} {'Best Val Acc':>10} {'Time (min)':>8} {'Status':>6}"
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary:
        logger.info(
            f"{row['model']:<24} {row['dataset']:<16} "
            f"{row['best_train_acc']:>12} {row['best_val_acc']:>10} {row['wall_time_min']:>8} {row['status']:>6}"
        )

def _setup_distributed():
    """
    Cluster runs launch via torchrun, which sets LOCAL_RANK and expects an
    NCCL process group. Local runs (`python -m src.main ...`, no torchrun)
    have neither: LOCAL_RANK is absent, there's no rendezvous, and calling
    init_process_group would hang/crash waiting for peers that don't exist.
    In that case, skip distributed init entirely and act as a single rank 0
    -- every `if local_rank == 0:` guard downstream then behaves exactly
    like a normal single-process run.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def _cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
