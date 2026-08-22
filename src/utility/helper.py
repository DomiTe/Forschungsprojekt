import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved FP32 models instead of training from scratch"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="FP32 baseline -> PTQ -> QAT training and checkpointing for all 3 models x both "
             "datasets, with throughput benchmarking at each stage. No Hessian/eigenvalue/quant-"
             "error/classification-metrics analysis -- run --checkpoint-metrics or "
             "--analyze-cifar10/--analyze-imagenet100 afterward for that, reading these same "
             "checkpoints. Writes results/<RUN_ID>/csv/{pipeline,ptq,qat}_summary.csv."
    )
    parser.add_argument(
        "--checkpoint-metrics",
        action="store_true",
        help="Per-layer Hessian trace, top eigenvalue, weight-quantization error (MSE/SQNR), and "
             "whole-model classification metrics for FP32/PTQ/QAT, computed from saved "
             "checkpoints (the analysis --train-only no longer runs inline). Writes "
             "results/<RUN_ID>/csv/{layerwise_hessian_traces,layerwise_top_eigenvalues,"
             "layerwise_quant_error,classification_metrics}.csv. Reuses --checkpoint-dir and "
             "--load-run-id. Prefers CUDA; runs as a single local process, no SLURM/torchrun "
             "required."
    )
    parser.add_argument(
        "--analyze-cifar10",
        action="store_true",
        help="Runs every retained analysis pipeline (checkpoint-metrics, relock-traces, "
             "quant-induced-trace, weight-ablation-canonical, spike-layer-cause, random-init-"
             "control, diagnose-activation-quant, ablate-layer-quantization) scoped to CIFAR10 "
             "only, all 3 models, reading checkpoints via --checkpoint-dir/--load-run-id (the "
             "checkpoints --train-only writes). Meant to run in parallel with "
             "--analyze-imagenet100 as a separate job/process. One step's failure is logged and "
             "does not abort the remaining steps. Excludes --weight-ablation-diagnose (a fixed "
             "one-off diagnostic, not a sweep) and --deploy-cpu-fbgemm/--diagnose-int8-perf "
             "(deployment/benchmark, not analysis) -- run those separately if needed."
    )
    parser.add_argument(
        "--analyze-imagenet100",
        action="store_true",
        help="Same bundle as --analyze-cifar10, scoped to IMAGENET100 only. Note: spike-layer-"
             "cause's cross-dataset resolution_contrast_ratio needs both datasets' data in the "
             "SAME process -- when run via --analyze-cifar10/--analyze-imagenet100 separately "
             "(e.g. in parallel), that one comparison stays NaN in both runs' "
             "spike_layer_attribution.csv; run --spike-layer-cause unscoped (both datasets, one "
             "process) separately to get it."
    )
    parser.add_argument(
        "--diagnose-int8-perf",
        action="store_true",
        help="Reconstruct fp32 and deployed int8 models fresh and diagnose why torchao "
             "int8 inference underperforms fp32 (throughput sweep + kernel profiling); "
             "writes results/<RUN_ID>/logs/int8_perf_diagnosis.txt (no training, no analysis)"
    )
    parser.add_argument(
        "--load-run-id",
        type=str,
        default=None,
        help="RUN_ID to load models from (defaults to current RUN_ID)"
    )
    parser.add_argument(
        "--deploy-cpu-fbgemm",
        action="store_true",
        help="Convert PoT-quantized PTQ/QAT checkpoints to real INT8 via torch.ao.quantization "
             "with the fbgemm backend, and benchmark accuracy/size/throughput on CPU. Intended "
             "for a local workstation run (python -m src.main ...), no SLURM/torchrun required."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to load PTQ/QAT checkpoints from for --deploy-cpu-fbgemm "
             "(e.g. results/backup_models/backup_quantized_models). When set, overrides "
             "--load-run-id; when unset, falls back to results/<load-run-id or RUN_ID>/quantized_models/."
    )
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=None,
        help="For --deploy-cpu-fbgemm: evaluate accuracy on only the "
             "first N batches of the validation set (useful for IMAGENET100 at 224x224 on CPU). "
             "Default: full validation set."
    )
    parser.add_argument(
        "--ablate-layer-quantization",
        action="store_true",
        help="Measure the accuracy impact of excluding individual layers from fbgemm INT8 "
             "quantization (leaving them in FP32), to test whether high-Hessian-trace layers "
             "cause the resnet50/CIFAR10/PTQ accuracy collapse. Writes "
             "results/<RUN_ID>/csv/layer_ablation.csv. Reuses --checkpoint-dir, --load-run-id "
             "and --eval-subset. Intended for a local workstation run (python -m src.main ...), "
             "no SLURM/torchrun required."
    )
    parser.add_argument(
        "--ablate-top-k",
        type=int,
        default=3,
        help="For --ablate-layer-quantization: number of highest-Hessian-trace layers to "
             "exclude from quantization (one at a time), plus the same number of "
             "lowest-nonzero-trace layers as a control (default: 3). Ignored when "
             "--ablate-layers is set."
    )
    parser.add_argument(
        "--ablate-layers",
        type=str,
        default=None,
        help="For --ablate-layer-quantization: comma-separated explicit layer names to "
             "exclude from quantization one at a time (e.g. 'conv1' or 'conv1,layer4.0.conv1'), "
             "instead of trace-guided top-k/low-k selection."
    )
    parser.add_argument(
        "--diagnose-activation-quant",
        action="store_true",
        help="Isolate how much accuracy damage comes from activation quantization versus weight "
             "quantization (resnet50/CIFAR10 PTQ: 80.56%% FP32 -> 11.73%% full-quantized, but "
             "75.46%% with only weight quantization -- most of the loss is activations, not "
             "weights), and identify which layers' activation ranges are pathological. Writes "
             "results/<RUN_ID>/csv/activation_{load_check,decomposition,ranges,ablation}.csv. "
             "Reuses --checkpoint-dir, --load-run-id and --eval-subset. Intended for a local "
             "workstation run (python -m src.main ...), no SLURM/torchrun required."
    )
    parser.add_argument(
        "--random-init-control",
        action="store_true",
        help="Compute the layer-wise weight-Hessian trace on an untrained (random-init) model "
             "and compare its per-layer profile against the trained-FP32 profile (recomputed in "
             "this run with an identical estimator config), to separate architectural curvature "
             "(already present at init) from learned curvature. CIFAR10; cnn, "
             "resnet18_no_weights, resnet50_no_weights (resnet50 first). Writes "
             "results/<RUN_ID>/csv/random_init_{traces,comparison,summary}.csv. Analysis only -- "
             "no quantization/PTQ/QAT/deployment. Reuses --checkpoint-dir (for the trained-FP32 "
             "baseline checkpoint) and --load-run-id. Prefers CUDA; runs as a single local "
             "process, no SLURM/torchrun required."
    )
    parser.add_argument(
        "--quant-induced-trace",
        action="store_true",
        help="Measure each layer's weight-Hessian trace across four model variants -- unfused "
             "FP32, fused FP32, PTQ, QAT -- with an identical estimator config and a single fixed "
             "probe seed, and decompose the FP32->quantized change into a fusion effect (BN "
             "folded into conv) and a quantization-induced effect (fake-quant noise). Includes a "
             "conv1 spotlight (trace across all variants, amplification, rank) and reconciliation "
             "against a banked FP32 profile (see --banked-fp32-profile). CIFAR10; "
             "resnet18_no_weights and resnet50_no_weights required, cnn included only if it fuses "
             "cleanly. Writes results/<RUN_ID>/csv/quant_induced_{traces,comparison,summary}.csv. "
             "Analysis only -- no torchao/INT8/deployment path. Reuses --checkpoint-dir and "
             "--load-run-id. Prefers CUDA; runs as a single local process, no SLURM/torchrun "
             "required."
    )
    parser.add_argument(
        "--banked-fp32-profile",
        type=str,
        default=None,
        help="Path to a banked FP32 Hessian-trace CSV (e.g. results/<RUN_ID>/csv/"
             "layerwise_hessian_traces.csv, or random_init_traces.csv) to reconcile against this "
             "run's freshly-computed fp32_unfused profile in --quant-induced-trace. Optional; "
             "omit to skip reconciliation (reported as banked_fp32_matches=not_provided)."
    )
    parser.add_argument(
        "--relock-traces",
        action="store_true",
        help="Freeze a single canonical Hessian-trace estimator configuration, diagnose which "
             "configuration knob produced each drifting legacy trace number (resnet50 conv1: "
             "13.87/11.79 FP32, 13161.8/1030.9 PTQ, ~14x elevation), recompute every headline "
             "trace from the frozen config, and write an old->new reconciliation ledger. Writes "
             "results/<RUN_ID>/trace_config.json and results/<RUN_ID>/csv/"
             "{drift_diagnosis,canonical_traces,trace_reconciliation_ledger}.csv. Analysis only. "
             "Reuses --checkpoint-dir and --load-run-id; see --banked-fp32-profile and "
             "--legacy-anchors for the ledger's reconciliation inputs. Prefers CUDA; runs as a "
             "single local process, no SLURM/torchrun required."
    )
    parser.add_argument(
        "--legacy-anchors",
        type=str,
        default=None,
        help="For --relock-traces: path to a JSON file (or an inline JSON string) overriding the "
             "built-in legacy anchor values (resnet50 conv1 FP32/PTQ numbers, elevation claim) "
             "that Part 1's drift-diagnosis grid tries to attribute to a configuration knob. "
             "Optional; omit to use the built-in defaults."
    )
    parser.add_argument(
        "--weight-ablation-canonical",
        action="store_true",
        help="Measure each layer's weight-only PoT quantization damage in isolation (P1, revised) "
             "and test whether it is predicted by the raw canonical weight-Hessian trace "
             "(fp32_fused, from --canonical-traces-csv), the weight-quantization perturbation "
             "||delta W||^2 alone, or the HAWQ product Tr(H)*||delta W||^2. CIFAR10; "
             "resnet18_no_weights and resnet50_no_weights required, cnn optional; PTQ required, "
             "QAT optional (skipped with a warning if its checkpoint is missing). Writes "
             "results/<RUN_ID>/csv/weight_ablation_canonical{,_correlation}.csv. Analysis only -- "
             "no torchao/deployment. Reuses --checkpoint-dir and --load-run-id; requires "
             "--canonical-traces-csv. Prefers CUDA; runs as a single local process, no "
             "SLURM/torchrun required."
    )
    parser.add_argument(
        "--canonical-traces-csv",
        type=str,
        default=None,
        help="For --weight-ablation-canonical: path to the canonical_traces.csv written by "
             "--relock-traces (results/<RUN_ID>/csv/canonical_traces.csv). Required -- the "
             "fp32_fused Tr(H) values used as the curvature term come from here, not recomputed."
    )
    parser.add_argument(
        "--damage-mode",
        type=str,
        default="both",
        choices=["signed", "abs", "both"],
        help="For --weight-ablation-canonical: which weight_damage_pts target the correlation "
             "summary's primary spearman_rho/spearman_p/top5_overlap/conv1_damage_rank columns "
             "are computed against -- 'abs' (|weight_damage_pts|, the paper's headline: PTQ "
             "damage is mostly positive and QAT damage is mostly negative, but both reflect the "
             "same sign-agnostic 'how much does this layer's quantization state matter', which "
             "conv1 tops in every configuration), 'signed' (the original P1-revised target, "
             "which reads as a null on QAT), or 'both' (default -- writes both column sets in "
             "the same row so neither is dropped). The per-layer CSV's weight_damage_pts stays "
             "signed regardless; abs_weight_damage_pts is added alongside it."
    )
    parser.add_argument(
        "--weight-ablation-loss",
        action="store_true",
        help="Extend --weight-ablation-canonical's isolation sweep with per-layer isolated "
             "validation LOSS (mean-reduced CrossEntropyLoss, same as train.py's _evaluate) "
             "alongside accuracy, across all {cnn,resnet18,resnet50}x{CIFAR10,IMAGENET100}x"
             "{PTQ,QAT} combinations (12 total), cheapest-first (CIFAR10 all models, then "
             "IMAGENET100 cnn/resnet18/resnet50). Reuses the isolation harness unchanged "
             "(_run_part0's gate, checkpoint loading, Identity-swap logic). Writes "
             "results/<RUN_ID>/csv/weight_ablation_loss_damage.csv one row at a time, flushed "
             "immediately -- resumable: re-launching skips (model,dataset,stage,layer) rows "
             "already present. Cross-validates each freshly recomputed accuracy against "
             "--existing-ablation-csv within 0.01pt and logs any drift to "
             "results/<RUN_ID>/csv/accuracy_mismatch.csv (always created, even empty) instead "
             "of silently accepting it. Requires --existing-ablation-csv (one per dataset). "
             "Analysis only -- no torchao/deployment. Reuses --checkpoint-dir/--load-run-id for "
             "model checkpoints. Prefers CUDA; single local process, no SLURM/torchrun required."
    )
    parser.add_argument(
        "--existing-ablation-csv",
        action="append",
        default=None,
        metavar="DATASET=PATH",
        help="For --weight-ablation-loss and --weight-ablation-loss-correlation: the canonical "
             "accuracy-only weight_ablation_canonical_v2.csv for one dataset, as DATASET=PATH "
             "(e.g. CIFAR10=results/20260816_230437_38678/csv/weight_ablation_canonical_v2.csv). "
             "Repeat once per dataset being processed. Supplies (a) the accuracy values the new "
             "loss sweep validates itself against, (b) the canonical per-layer set used to "
             "determine when a combo is fully done (resumability) and when a combo has all its "
             "layers (Part 5 completeness), and (c) the S_raw/S_pert/S_hawq predictor columns "
             "the loss-based correlation step needs."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="For --weight-ablation-loss: ignore any existing weight_ablation_loss_damage.csv "
             "progress and recompute every row from scratch. The existing output CSV and "
             "accuracy_mismatch.csv are backed up (renamed with a timestamp suffix), never "
             "deleted. Default off -- normal re-launches resume instead."
    )
    parser.add_argument(
        "--weight-ablation-loss-correlation",
        action="store_true",
        help="Separate, fast, independently-rerunnable step (Part 5): correlates S_raw/S_pert/"
             "S_hawq against loss-based damage (abs_loss_damage/loss_damage) using whatever rows "
             "currently exist in --loss-damage-csv (safe on a partial/still-running sweep). "
             "Mirrors weight_ablation_canonical_correlation_v2.csv's schema exactly (same "
             "_abs/_signed columns) plus a status column -- (model,dataset,stage) combos missing "
             "layers are marked status=incomplete_N_of_TOTAL with correlation fields left blank, "
             "never computed on a truncated subset. Requires --existing-ablation-csv (for the "
             "predictor columns and each combo's canonical layer count). Writes "
             "results/<RUN_ID>/csv/weight_ablation_loss_damage_correlation.csv. No model/GPU "
             "involved -- runs in seconds."
    )
    parser.add_argument(
        "--loss-damage-csv",
        type=str,
        default=None,
        help="For --weight-ablation-loss-correlation: path to weight_ablation_loss_damage.csv. "
             "Defaults to results/<RUN_ID>/csv/weight_ablation_loss_damage.csv (current RUN_ID)."
    )
    parser.add_argument(
        "--loss-correlation-output-csv",
        type=str,
        default=None,
        help="For --weight-ablation-loss-correlation: output path. Defaults to "
             "results/<RUN_ID>/csv/weight_ablation_loss_damage_correlation.csv (current RUN_ID)."
    )
    parser.add_argument(
        "--weight-ablation-diagnose",
        action="store_true",
        help="Diagnose the v1->v2 weight-ablation-canonical damage-magnitude drift (resnet50 "
             "conv1 PTQ: 0.67 -> 1.62pts, ~2.4x; resnet18 conv1 PTQ: same direction/rough factor) "
             "by varying one candidate cause at a time -- eval batch size, num_workers/shuffle, "
             "seed/dtype/determinism, the weight_fake_quant isolation mechanism, the FP32/PTQ "
             "checkpoint source, fresh-per-layer model reconstruction, BN eval mode, CUDA "
             "nondeterminism -- from the v2 configuration on resnet50/CIFAR10/PTQ conv1 (the "
             "cleanest anchor), holding everything else fixed, to identify which flip reproduces "
             "the v1 magnitude. Writes results/<RUN_ID>/csv/weight_ablation_drift_ledger.csv. "
             "Analysis only. Reuses --checkpoint-dir and --load-run-id for the v2 checkpoint set; "
             "see --diagnose-v1-checkpoint-dir for the v1 checkpoint set being compared against. "
             "Prefers CUDA; runs as a single local process, no SLURM/torchrun required."
    )
    parser.add_argument(
        "--diagnose-v1-checkpoint-dir",
        type=str,
        default=None,
        help="For --weight-ablation-diagnose: directory holding the v1 quantized checkpoints "
             "(candidate 5 -- 'which checkpoint set is being differenced'), with its FP32 "
             "baseline resolved as the sibling 'models' directory (same convention as "
             "--checkpoint-dir elsewhere). Defaults to results/backup_models/quantized_models, "
             "the checkpoint set the original v1 investigation run resolved to."
    )
    parser.add_argument(
        "--spike-layer-cause",
        action="store_true",
        help="Identify the 'spike layer' (highest fused-basis Tr(H); highest |weight_damage_pts|) "
             "per model x dataset -- no hardcoded conv1 -- then attribute its size-independent "
             "curvature excess (log-log residual above the trace_per_param-vs-numel fit) to "
             "architectural descriptors (fan_in, output_map, Tr(A_prev)), using a KFAC-style "
             "patch-unfolded forward/backward (A/G) hook split to localise the excess, and a "
             "CIFAR10 (32x32) vs IMAGENET100 (224x224) resolution contrast to discriminate spatial-"
             "extent (H1), fan-in (H2), and input-covariance-conditioning (H3) hypotheses. Models "
             "cnn, resnet18_no_weights, resnet50_no_weights; variants fp32_unfused, fp32_fused, "
             "ptq (QAT excluded -- a training regime, not an architectural property). Writes "
             "results/<RUN_ID>/csv/spike_{selection,layer_traces,layer_residual,layer_descriptors,"
             "layer_kfac,layer_attribution}.csv. Analysis only -- no torchao/deployment. Reuses "
             "--checkpoint-dir, --load-run-id, and --canonical-traces-csv (extends its sibling "
             "trace_config.json with an IMAGENET100 entry; CIFAR10 entry untouched). Prefers CUDA "
             "(A100); runs as a single local process, no SLURM/torchrun required. NOT run under "
             "torch.no_grad() -- the KFAC backward-hook measurement needs real gradients."
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Global reproducibility knob: for the phases whose outputs depend on Hutchinson probe "
             "seeds or val-batch selection (--relock-traces, --random-init-control, "
             "--quant-induced-trace, --diagnose-activation-quant's Part 2 activation ranges, "
             "--spike-layer-cause), run the core stochastic measurement N times under distinct "
             "derived seeds ([--base-seed, --base-seed+1, ..., --base-seed+N-1]) and emit mean +/- "
             "std across seeds (an additive 'seed'/'n_seeds'/'metric_std' CSV column triple; "
             "per-seed rows are always kept, never destructively aggregated). Default 1 -- matches "
             "the ships-headline-numbers config and (mechanism-for-mechanism) reproduces each "
             "phase's pre-n_seeds behavior; a bare --n-seeds 1 will only numerically match an old "
             "saved CSV if --base-seed is also set to that phase's historical constant -- each "
             "affected phase logs a warning naming the exact flags when this isn't the case. "
             "Phases NOT in this list (--weight-ablation-canonical, --diagnose-activation-quant "
             "Parts 1/3, --deploy-cpu-fbgemm) are deterministic on a fixed model/checkpoint and are "
             "left untouched -- no seed columns, no repeated computation. Analysis only; does not "
             "retrain or resample any model."
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Starting seed for --n-seeds' derived seed list ([--base-seed, ..., --base-seed+"
             "n_seeds-1]). Default 42. See --n-seeds for which phases this affects and the "
             "historical-constant caveat at --n-seeds 1."
    )
    parser.add_argument(
        "--imagenet100-checkpoint-dir",
        type=str,
        default=None,
        help="For --spike-layer-cause: overrides the quantized-checkpoint directory (FP32 baseline "
             "resolved as its sibling 'models' dir, same convention as --checkpoint-dir) used for "
             "IMAGENET100 only -- CIFAR10 always resolves via --checkpoint-dir/--load-run-id. No "
             "single run_id in this project currently banks fresh checkpoints for both datasets at "
             "once, so this lets the cross-dataset comparison pull each dataset from its own run "
             "without silently mixing checkpoint provenance within a single dataset's numbers. "
             "Omit to use the same source (--checkpoint-dir/--load-run-id) for both datasets."
    )
    return parser.parse_args()