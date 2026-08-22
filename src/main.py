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

from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.top_eigenvalue import compute_top_eigenvalue
from src.analysis.quant_error import compute_layerwise_quant_error
from src.analysis.classification_metrics import compute_classification_metrics
from src.analysis import int8_profile

from src.utility.helper import parse_args
from src.utility.utils import setup_global_logging, get_data_loaders, measure_throughput
from src.utility.config import (
    CSV_DIR,
    LOG_DIR,
    BASE_DIR,
    DATASET_SPECS,
    QUANTIZED_MODELS,
    # DEPLOYED_MODELS,
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
    "IMAGENET100",
]


def _parse_dataset_path_pairs(pairs: list[str] | None, flag_name: str) -> dict[str, str]:
    # Parses repeated "DATASET=PATH" CLI values (e.g. --existing-ablation-csv
    # CIFAR10=... --existing-ablation-csv IMAGENET100=...) into a dict --
    # used by --weight-ablation-loss / --weight-ablation-loss-correlation,
    # which each need one canonical accuracy-only CSV per dataset (those
    # live in separate per-dataset results/<RUN_ID>/ directories, not one
    # shared file).
    if not pairs:
        raise ValueError(f"{flag_name} is required (one DATASET=PATH per dataset being processed)")
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{flag_name} value {pair!r} is not in DATASET=PATH form")
        dataset_name, path = pair.split("=", 1)
        result[dataset_name] = path
    return result


def main() -> None:
    args = parse_args()
    local_rank = _setup_distributed()
    total = len(MODELS) * len(DATASETS)
    
    if local_rank == 0:
        setup_global_logging()
        
        logger.info(f"=== Pipeline start: {len(MODELS)} models - {len(DATASETS)} datasets = {total} runs ===")

    # -------------------------------------------------------------------
    # Train-Only mode: FP32 -> PTQ -> QAT training and checkpointing for
    # all 3 models x both datasets, with throughput benchmarking. No
    # Hessian/eigenvalue/quant-error/classification analysis -- see
    # --checkpoint-metrics / --analyze-{dataset} for that, run afterward
    # from these same checkpoints.
    # -------------------------------------------------------------------
    if args.train_only:
        if local_rank == 0:
            logger.info("=== Train-Only: FP32/PTQ/QAT training only, no analysis ===")
        _run_train_only(args, local_rank)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Checkpoint-Metrics mode: per-layer Hessian trace, top eigenvalue,
    # weight-quantization error, and classification metrics for FP32/PTQ/
    # QAT, computed from saved checkpoints. All logic lives in
    # src/analysis/checkpoint_metrics.py. Skips training entirely. Runs as
    # a single local process (no torchrun/distributed init needed).
    # -------------------------------------------------------------------
    if args.checkpoint_metrics:
        if local_rank == 0:
            logger.info("=== Checkpoint-Metrics: skipping training ===")
            from src.analysis.checkpoint_metrics import run_checkpoint_metrics
            run_checkpoint_metrics(checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id)
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Analyze-CIFAR10 / Analyze-IMAGENET100 modes: every retained analysis
    # pipeline (see ANALYZE_STEPS / _run_analyze_dataset), scoped to one
    # dataset, all 3 models, so the two can run as separate parallel jobs.
    # Skips training entirely.
    # -------------------------------------------------------------------
    if args.analyze_cifar10:
        _run_analyze_dataset(args, local_rank, "CIFAR10")
        _cleanup()
        return

    if args.analyze_imagenet100:
        _run_analyze_dataset(args, local_rank, "IMAGENET100")
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
    # Ablate-Layer-Quantization mode: measure the accuracy impact of
    # excluding individual layers from fbgemm INT8 quantization, to test
    # whether high-Hessian-trace layers cause the resnet50/CIFAR10/PTQ
    # accuracy collapse. All logic lives in src/analysis/layer_ablation.py,
    # which reuses (does not duplicate) the checkpoint reconstruction and
    # fbgemm conversion path from src/quantization/deploy_fbgemm.py. Skips
    # FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis. Runs
    # as a single local process (no torchrun/distributed init needed).
    # -------------------------------------------------------------------
    if args.ablate_layer_quantization:
        if local_rank == 0:
            logger.info("=== Ablate-Layer-Quantization: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.layer_ablation import run_layer_ablation
        run_layer_ablation(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            ablate_top_k=args.ablate_top_k,
            ablate_layers=args.ablate_layers,
            eval_subset=args.eval_subset,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Diagnose-Activation-Quant mode: isolate how much accuracy damage comes
    # from activation quantization versus weight quantization, and identify
    # which layers' activation ranges are pathological. All logic lives in
    # src/analysis/diagnose_activations.py, which reuses (does not duplicate)
    # the checkpoint reconstruction primitives from
    # src/quantization/quantizer.py and src/quantization/deploy_fbgemm.py.
    # Skips FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis.
    # Runs as a single local process (no torchrun/distributed init needed).
    # -------------------------------------------------------------------
    if args.diagnose_activation_quant:
        if local_rank == 0:
            logger.info("=== Diagnose-Activation-Quant: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.diagnose_activations import run_diagnose_activation_quant
        run_diagnose_activation_quant(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            eval_subset=args.eval_subset,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Random-Init-Control mode: compute the layer-wise weight-Hessian trace
    # on an untrained (random-init) model and compare its per-layer profile
    # against the trained-FP32 profile (recomputed in this same run with an
    # identical estimator config), to separate architectural curvature from
    # learned curvature. All logic lives in
    # src/analysis/random_init_control.py, which reuses (does not
    # duplicate) compute_layerwise_hessian_trace_pyhessian and the FP32
    # checkpoint resolution helpers from src/analysis/diagnose_activations.py.
    # Analysis only -- no quantization/PTQ/QAT/deployment. Skips FP32/PTQ/QAT
    # training and all Hessian/eigenvalue/SQNR analysis. Runs as a single
    # local process (no torchrun/distributed init needed), prefers CUDA.
    # -------------------------------------------------------------------
    if args.random_init_control:
        if local_rank == 0:
            logger.info("=== Random-Init-Control: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.random_init_control import run_random_init_control
        run_random_init_control(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Quant-Induced-Trace mode: measure each layer's weight-Hessian trace
    # across four model variants (unfused FP32, fused FP32, PTQ, QAT) with an
    # identical estimator configuration, and decompose the FP32->quantized
    # change into a fusion effect and a quantization-induced effect. All
    # logic lives in src/analysis/quant_induced_trace.py, which reuses (does
    # not duplicate) the loader chain (build_model -> fuse_model_architectures
    # -> replace_layers_for_quantization -> load_state_dict),
    # compute_layerwise_hessian_trace_pyhessian, and the Identity-swap helper
    # from src/analysis/diagnose_activations.py. Analysis only -- no
    # torchao/INT8/deployment. Skips FP32/PTQ/QAT training and all
    # Hessian/eigenvalue/SQNR analysis. Runs as a single local process (no
    # torchrun/distributed init needed), prefers CUDA.
    # -------------------------------------------------------------------
    if args.quant_induced_trace:
        if local_rank == 0:
            logger.info("=== Quant-Induced-Trace: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.quant_induced_trace import run_quant_induced_trace
        run_quant_induced_trace(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            banked_fp32_profile=args.banked_fp32_profile,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Relock-Traces mode: freeze a single canonical Hessian-trace estimator
    # configuration, diagnose which configuration knob produced each
    # drifting legacy trace number (a one-knob-at-a-time grid on resnet50
    # conv1), recompute every headline trace from the frozen config, and
    # write an old->new reconciliation ledger. All logic lives in
    # src/analysis/relock_traces.py, which reuses (does not duplicate)
    # compute_layerwise_hessian_trace_pyhessian and the quant-induced mode's
    # Part 0 name/shape mapping gate and model-construction helpers
    # (src/analysis/quant_induced_trace.py). Analysis only. Skips
    # FP32/PTQ/QAT training and all Hessian/eigenvalue/SQNR analysis. Runs
    # as a single local process (no torchrun/distributed init needed),
    # prefers CUDA.
    # -------------------------------------------------------------------
    if args.relock_traces:
        if local_rank == 0:
            logger.info("=== Relock-Traces: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.relock_traces import run_relock_traces
        run_relock_traces(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            banked_fp32_profile=args.banked_fp32_profile,
            legacy_anchors=args.legacy_anchors,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Weight-Ablation-Canonical mode: measure each layer's weight-only PoT
    # quantization damage in isolation (P1, revised) and test whether it is
    # predicted by the raw canonical weight-Hessian trace, the weight-quant
    # perturbation ||delta W||^2 alone, or the HAWQ product Tr(H)*||delta
    # W||^2. All logic lives in src/analysis/weight_ablation_canonical.py,
    # which reuses (does not duplicate) the loader, bake_pot_into_standard_
    # layers, the evaluation function, and the Identity-swap helper from
    # src/analysis/diagnose_activations.py, plus P1's path-equivalence gate
    # and robust checkpoint resolver from src/analysis/weight_ablation.py.
    # Analysis only -- no torchao/deployment. Skips FP32/PTQ/QAT training
    # and all Hessian/eigenvalue/SQNR analysis. Runs as a single local
    # process (no torchrun/distributed init needed), prefers CUDA.
    # -------------------------------------------------------------------
    if args.weight_ablation_canonical:
        if local_rank == 0:
            logger.info("=== Weight-Ablation-Canonical: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.weight_ablation_canonical import run_weight_ablation_canonical
        run_weight_ablation_canonical(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            canonical_traces_csv=args.canonical_traces_csv,
            damage_mode=args.damage_mode,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Weight-Ablation-Loss mode: extends the isolation sweep above with
    # per-layer isolated validation LOSS alongside accuracy (both
    # supervisors asked whether the damage metric should be loss- rather
    # than accuracy-based). Incremental, resumable -- see
    # src/analysis/weight_ablation_canonical.py's "Loss-based damage
    # extension" section for the full design. Analysis only; reuses the
    # accuracy-only isolation harness unchanged.
    # -------------------------------------------------------------------
    if args.weight_ablation_loss:
        if local_rank == 0:
            logger.info("=== Weight-Ablation-Loss: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.weight_ablation_canonical import run_weight_ablation_loss
        existing_ablation_csvs = _parse_dataset_path_pairs(args.existing_ablation_csv, "--existing-ablation-csv")
        run_weight_ablation_loss(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            existing_ablation_csvs=existing_ablation_csvs,
            force_recompute=args.force_recompute,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Weight-Ablation-Loss-Correlation mode: Part 5 of the loss extension --
    # separate, fast, GPU-free, safe on partial data. See
    # src/analysis/weight_ablation_canonical.py's run_weight_ablation_loss_correlation.
    # -------------------------------------------------------------------
    if args.weight_ablation_loss_correlation:
        if local_rank == 0:
            logger.info("=== Weight-Ablation-Loss-Correlation: no model/GPU involved ===")
        from src.analysis.weight_ablation_canonical import run_weight_ablation_loss_correlation
        existing_ablation_csvs = _parse_dataset_path_pairs(args.existing_ablation_csv, "--existing-ablation-csv")
        loss_damage_csv = args.loss_damage_csv or os.path.join(CSV_DIR, "weight_ablation_loss_damage.csv")
        output_csv = args.loss_correlation_output_csv or os.path.join(CSV_DIR, "weight_ablation_loss_damage_correlation.csv")
        run_weight_ablation_loss_correlation(
            loss_damage_csv=loss_damage_csv,
            existing_ablation_csvs=existing_ablation_csvs,
            output_csv=output_csv,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Weight-Ablation-Diagnose mode: explain the v1->v2 weight-ablation-
    # canonical damage-magnitude drift (resnet50/CIFAR10/PTQ conv1: 0.67 ->
    # 1.62pts, ~2.4x) by varying one candidate cause at a time -- eval batch
    # size, num_workers/shuffle, seed/dtype/determinism, the weight_fake_quant
    # isolation mechanism, the FP32/PTQ checkpoint source, fresh-per-layer
    # model reconstruction, BN eval mode, CUDA nondeterminism -- from the v2
    # configuration on resnet50/CIFAR10/PTQ conv1, holding everything else
    # fixed. All logic lives in src/analysis/weight_ablation_diagnose.py,
    # which reuses (does not duplicate) the checkpoint loader, Identity-swap
    # helpers, weight-mask verifier and robust checkpoint resolver already
    # established by weight_ablation.py / diagnose_activations.py. Analysis
    # only -- no torchao/deployment. Skips FP32/PTQ/QAT training and all
    # Hessian/eigenvalue/SQNR analysis. Runs as a single local process (no
    # torchrun/distributed init needed), prefers CUDA.
    # -------------------------------------------------------------------
    if args.weight_ablation_diagnose:
        if local_rank == 0:
            logger.info("=== Weight-Ablation-Diagnose: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.weight_ablation_diagnose import run_weight_ablation_diagnose
        run_weight_ablation_diagnose(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            v1_checkpoint_dir=args.diagnose_v1_checkpoint_dir,
        )
        _cleanup()
        return

    # -------------------------------------------------------------------
    # Spike-Layer-Cause mode: identifies the "spike layer" (highest fused-
    # basis Tr(H); highest |weight_damage_pts|) per model x dataset -- no
    # hardcoded conv1 -- then attributes its size-independent curvature
    # excess to architectural descriptors (fan_in, output_map, Tr(A_prev))
    # via a KFAC-style patch-unfolded forward/backward (A/G) hook split, and
    # a CIFAR10-vs-IMAGENET100 resolution contrast to discriminate spatial-
    # extent (H1), fan-in (H2), and input-covariance-conditioning (H3)
    # hypotheses. All logic lives in src/analysis/spike_layer_cause.py,
    # which reuses (does not duplicate) compute_layerwise_hessian_trace_
    # pyhessian, the quant-induced mode's Part 0 mapping gate and model-
    # construction helpers, and weight_ablation_canonical.py's own isolation
    # sweep for the damage-based spike selection. Analysis only -- no
    # torchao/deployment. Skips FP32/PTQ/QAT training and the eigenvalue/
    # SQNR analyses. Runs as a single local process (no torchrun/distributed
    # init needed), prefers CUDA. NOT run under torch.no_grad() -- the KFAC
    # backward-hook measurement needs real gradients.
    # -------------------------------------------------------------------
    if args.spike_layer_cause:
        if local_rank == 0:
            logger.info("=== Spike-Layer-Cause: skipping training and Hessian/eigenvalue/SQNR analysis ===")
        from src.analysis.spike_layer_cause import run_spike_layer_cause
        run_spike_layer_cause(
            checkpoint_dir=args.checkpoint_dir,
            load_run_id=args.load_run_id,
            canonical_traces_csv=args.canonical_traces_csv,
            imagenet100_checkpoint_dir=args.imagenet100_checkpoint_dir,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
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



def _run_train_only(args, local_rank: int) -> None:
    """
    FP32 baseline -> PTQ -> QAT training and checkpointing, with throughput
    benchmarking at each stage, and NO Hessian/eigenvalue/quant-error/
    classification-metrics analysis -- that analysis reads these same
    checkpoints afterward via --checkpoint-metrics (see
    src/analysis/checkpoint_metrics.py) or the --analyze-{dataset} bundles,
    so it can be rerun independently without retraining. This is the
    default (no-flag) pipeline's training logic with every analysis call
    removed; the default pipeline itself is left untouched for backward
    compatibility with existing invocations that expect the old combined
    behavior.
    """
    total = len(MODELS) * len(DATASETS)
    summary: list[dict] = []
    ptq_summary: list[dict] = []
    qat_summary: list[dict] = []
    run_idx = 0

    for dataset_name in DATASETS:
        if local_rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"[TrainOnly] Loading dataset: {dataset_name}")
        try:
            specs = DATASET_SPECS[dataset_name]
            train_loader, val_loader, num_classes = get_data_loaders(dataset_name)
        except Exception as exc:
            if local_rank == 0:
                logger.error(f"[TrainOnly] Failed to load {dataset_name}: {exc}")
            for model_name in MODELS:
                summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "best_val_acc": "LOAD_ERROR", "wall_time_min": "", "status": "failed",
                })
            continue

        for model_name in MODELS:
            run_idx += 1
            if local_rank == 0:
                logger.info(f"\n--- [TrainOnly] Run {run_idx}/{total}: {model_name} on {dataset_name} ---")

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
                        BASE_DIR, "results", load_run_id, "models",
                        f"baseline_{model_name}_{dataset_name}_float32.pt"
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"No saved model at {model_path}")
                    fp32_model = build_model(
                        num_classes=specs["num_classes"], model_name=model_name,
                        channels=specs["channels"], image_size=specs["image_size"]
                    ).to(device)
                    fp32_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                    _, best_val_acc = _evaluate(fp32_model, val_loader, torch.nn.CrossEntropyLoss(), device)
                    history = {"train_acc": [0.0], "val_acc": [best_val_acc]}
                else:
                    fp32_model, history, _ = train_model(
                        train_loader, val_loader, num_classes,
                        model_name=model_name, dataset_name=dataset_name,
                    )

                device = next(fp32_model.parameters()).device
                best_val_acc = max(history["val_acc"])
                unwrapped_fp32 = fp32_model.module if hasattr(fp32_model, "module") else fp32_model

                dummy_shape = (1, specs["channels"], specs["image_size"], specs["image_size"])
                fp32_metrics = measure_throughput(fp32_model, device, dummy_shape)
                if local_rank == 0:
                    logger.info(f"[TrainOnly][FP32] Best Val Acc: {best_val_acc:.2f}% | Latency: {fp32_metrics['latency_ms']:.2f}ms")

                fp32_model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # -------------------------------------------------------------
                # PTQ Workflow (Power-of-Two + Asymmetric)
                # -------------------------------------------------------------
                if local_rank == 0:
                    logger.info("[TrainOnly] Starting PTQ Calibration...")

                ptq_model = copy.deepcopy(unwrapped_fp32)  # preserve FP32 weights for QAT below
                del fp32_model
                torch.cuda.empty_cache()

                ptq_model.eval()
                fuse_model_architectures(ptq_model, model_name)
                replace_layers_for_quantization(ptq_model)
                ptq_model = ptq_model.to(device)
                calibrate_ptq(ptq_model, train_loader, device, num_batches=20)

                ptq_loss, ptq_acc = _evaluate(ptq_model, val_loader, torch.nn.CrossEntropyLoss(), device)

                if local_rank == 0:
                    ptq_path = os.path.join(QUANTIZED_MODELS, f"ptq_po2_{model_name}_{dataset_name}.pt")
                    torch.save(ptq_model.state_dict(), ptq_path)
                    logger.info(f"[TrainOnly] Saved PTQ state_dict -> {ptq_path}")

                # max-autotune profiles Triton kernels on the target GPU to find the fastest one
                compiled_ptq_model = torch.compile(ptq_model, mode="max-autotune")
                ptq_metrics = measure_throughput(compiled_ptq_model, device, dummy_shape)

                if local_rank == 0:
                    logger.info(f"[TrainOnly][PTQ] Val Acc: {ptq_acc:.2f}% | Latency: {ptq_metrics['latency_ms']:.2f}ms")
                    ptq_summary.append({
                        "model": model_name, "dataset": dataset_name,
                        "fp32_val_acc": f"{best_val_acc:.2f}", "ptq_val_acc": f"{ptq_acc:.2f}",
                        "acc_drop": f"{best_val_acc - ptq_acc:.2f}",
                        "fp32_fps": f"{fp32_metrics['throughput_fps']:.1f}", "ptq_fps": f"{ptq_metrics['throughput_fps']:.1f}",
                        "speedup": f"{ptq_metrics['throughput_fps'] / fp32_metrics['throughput_fps']:.2f}",
                        "status": "ok",
                    })

                compiled_ptq_model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # -------------------------------------------------------------
                # QAT Workflow (Building on PTQ)
                # -------------------------------------------------------------
                if local_rank == 0:
                    logger.info("[TrainOnly] Starting Quantization-Aware Training (QAT)...")

                qat_model, qat_history, _ = train_qat(
                    ptq_model=ptq_model, train_loader=train_loader, val_loader=val_loader,
                    device=device, epochs=QAT_EPOCH, lr=QAT_LR,
                )
                qat_acc = max(qat_history["val_acc"])

                if local_rank == 0:
                    qat_path = os.path.join(QUANTIZED_MODELS, f"qat_po2_{model_name}_{dataset_name}.pt")
                    torch.save(qat_model.state_dict(), qat_path)
                    logger.info(f"[TrainOnly] Saved QAT state_dict -> {qat_path}")

                compiled_qat_model = torch.compile(qat_model, mode="max-autotune")
                qat_metrics = measure_throughput(compiled_qat_model, device, dummy_shape)

                if local_rank == 0:
                    logger.info(f"[TrainOnly][QAT] Val Acc: {qat_acc:.2f}% | Latency: {qat_metrics['latency_ms']:.2f}ms")
                    qat_summary.append({
                        "model": model_name, "dataset": dataset_name,
                        "ptq_val_acc": f"{ptq_acc:.2f}", "qat_val_acc": f"{qat_acc:.2f}",
                        "acc_recovered": f"{qat_acc - ptq_acc:.2f}",
                        "fp32_fps": f"{fp32_metrics['throughput_fps']:.1f}", "qat_fps": f"{qat_metrics['throughput_fps']:.1f}",
                        "speedup": f"{qat_metrics['throughput_fps'] / fp32_metrics['throughput_fps']:.2f}",
                        "status": "ok",
                    })

                elapsed_min = (time.perf_counter() - t0) / 60
                summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "best_train_acc": f"{max(history['train_acc']):.2f}", "best_val_acc": f"{best_val_acc:.2f}",
                    "ptq_val_acc": f"{ptq_acc:.2f}", "qat_val_acc": f"{qat_acc:.2f}",
                    "fp32_fps": f"{fp32_metrics['throughput_fps']:.1f}", "ptq_fps": f"{ptq_metrics['throughput_fps']:.1f}",
                    "qat_fps": f"{qat_metrics['throughput_fps']:.1f}",
                    "wall_time_min": f"{elapsed_min:.1f}", "status": "ok",
                })

                compiled_qat_model.zero_grad(set_to_none=True)
                del compiled_ptq_model, compiled_qat_model, ptq_model, qat_model
                torch.cuda.empty_cache()

            except Exception as exc:
                elapsed_min = (time.perf_counter() - t0) / 60
                if local_rank == 0:
                    logger.error(f"[TrainOnly] FAILED {model_name}/{dataset_name}: {exc}", exc_info=True)
                summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "best_train_acc": "ERROR", "best_val_acc": "ERROR", "ptq_val_acc": "ERROR", "qat_val_acc": "ERROR",
                    "fp32_fps": "ERROR", "ptq_fps": "ERROR", "qat_fps": "ERROR",
                    "wall_time_min": f"{elapsed_min:.1f}", "status": "failed",
                })
                ptq_summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "fp32_val_acc": "ERROR", "ptq_val_acc": "ERROR", "acc_drop": "ERROR",
                    "fp32_fps": "ERROR", "ptq_fps": "ERROR", "speedup": "ERROR", "status": "failed",
                })
                qat_summary.append({
                    "model": model_name, "dataset": dataset_name,
                    "ptq_val_acc": "ERROR", "qat_val_acc": "ERROR", "acc_recovered": "ERROR",
                    "fp32_fps": "ERROR", "qat_fps": "ERROR", "speedup": "ERROR", "status": "failed",
                })
            finally:
                import gc
                for name in ['fp32_model', 'ptq_model', 'qat_model', 'compiled_ptq_model', 'compiled_qat_model']:
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
        _print_summary(summary)
        logger.info("=== Train-Only complete ===")


# Order matters: checkpoint-metrics writes layerwise_hessian_traces.csv,
# which --ablate-layer-quantization (layer_ablation.py) reads for its
# trace-guided top-k/low-k layer selection; relock-traces writes
# canonical_traces.csv, which weight-ablation-canonical and spike-layer-
# cause both read (via canonical_traces_csv) for their fused-basis Tr(H)
# lookups. Both must run before their respective consumers.
ANALYZE_STEPS = [
    "checkpoint-metrics", "relock-traces", "quant-induced-trace",
    "weight-ablation-canonical", "spike-layer-cause", "random-init-control",
    "diagnose-activation-quant", "ablate-layer-quantization",
]


def _run_analyze_dataset(args, local_rank: int, dataset_name: str) -> None:
    """
    Runs every retained analysis pipeline (ANALYZE_STEPS) scoped to ONE
    dataset, all 3 models, reading checkpoints via --checkpoint-dir/
    --load-run-id (the same checkpoints --train-only writes). Scoped to one
    dataset so --analyze-cifar10 and --analyze-imagenet100 can run as
    separate, parallel jobs. One step's failure is logged and does not
    abort the remaining steps -- this is a multi-hour bundle, not a single
    atomic operation.

    Not bundled here: --weight-ablation-diagnose (a fixed one-off
    diagnostic, not a per-model-per-dataset sweep -- its drift question is
    already answered) and --deploy-cpu-fbgemm/--diagnose-int8-perf
    (deployment/benchmark concerns, not analysis).
    """
    from src.analysis.checkpoint_metrics import run_checkpoint_metrics
    from src.analysis.relock_traces import run_relock_traces
    from src.analysis.quant_induced_trace import run_quant_induced_trace
    from src.analysis.weight_ablation_canonical import run_weight_ablation_canonical
    from src.analysis.spike_layer_cause import run_spike_layer_cause
    from src.analysis.random_init_control import run_random_init_control
    from src.analysis.diagnose_activations import run_diagnose_activation_quant
    from src.analysis.layer_ablation import run_layer_ablation

    if local_rank != 0:
        return

    datasets = [dataset_name]
    label = f"Analyze-{dataset_name}"
    logger.info(f"=== {label}: {len(ANALYZE_STEPS)} pipelines, all models, {dataset_name} only ===")

    def _step(n: int, name: str, fn, **kwargs) -> None:
        logger.info(f"[{label}] --- {n}/{len(ANALYZE_STEPS)}: {name} ---")
        try:
            fn(**kwargs)
        except Exception as exc:
            logger.error(f"[{label}] {name} FAILED -- {exc}", exc_info=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _step(1, "checkpoint-metrics", run_checkpoint_metrics,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id, datasets=datasets)

    _step(2, "relock-traces", run_relock_traces,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          banked_fp32_profile=args.banked_fp32_profile, legacy_anchors=args.legacy_anchors, datasets=datasets,
          n_seeds=args.n_seeds, base_seed=args.base_seed)
    canonical_traces_csv = args.canonical_traces_csv or os.path.join(CSV_DIR, "canonical_traces.csv")

    _step(3, "quant-induced-trace", run_quant_induced_trace,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          banked_fp32_profile=args.banked_fp32_profile, datasets=datasets,
          n_seeds=args.n_seeds, base_seed=args.base_seed)

    _step(4, "weight-ablation-canonical", run_weight_ablation_canonical,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          canonical_traces_csv=canonical_traces_csv, damage_mode=args.damage_mode, datasets=datasets)

    _step(5, "spike-layer-cause", run_spike_layer_cause,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          canonical_traces_csv=canonical_traces_csv,
          imagenet100_checkpoint_dir=args.imagenet100_checkpoint_dir, datasets=datasets,
          n_seeds=args.n_seeds, base_seed=args.base_seed)

    _step(6, "random-init-control", run_random_init_control,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id, datasets=datasets,
          n_seeds=args.n_seeds, base_seed=args.base_seed)

    _step(7, "diagnose-activation-quant", run_diagnose_activation_quant,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          eval_subset=args.eval_subset, datasets=datasets,
          n_seeds=args.n_seeds, base_seed=args.base_seed)

    _step(8, "ablate-layer-quantization", run_layer_ablation,
          checkpoint_dir=args.checkpoint_dir, load_run_id=args.load_run_id,
          ablate_top_k=args.ablate_top_k, ablate_layers=args.ablate_layers,
          eval_subset=args.eval_subset, datasets=datasets)

    logger.info(f"=== {label} complete ===")


DIAGNOSE_INT8_PERF_DATASETS = ["CIFAR10", "IMAGENET100"]

# (stage label, checkpoint filename prefix) -- both PTQ and QAT checkpoints
# went through fuse_model_architectures + replace_layers_for_quantization,
# so they share the same custom-quantized-layer structure and loader.
DIAGNOSE_INT8_PERF_STAGES = [
    ("PTQ", "ptq_po2"),
    ("QAT", "qat_po2"),
]


def _run_diagnose_int8_perf(args, local_rank: int) -> None:
    """
    Orchestrates int8_profile's fp32-vs-int8 performance diagnosis across
    the model/dataset/stage matrix. All reconstruction/benchmarking/
    profiling logic lives in src.analysis.int8_profile; this just loops,
    collects results, and writes the sweep CSV + combined text report.

    Needs no data loader and no DDP-wrapped forward pass (synthetic
    fixed-shape inputs only), so the whole body is
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
