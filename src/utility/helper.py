import argparse

def parse_args():
    # Deferred so this module keeps importing nothing heavy at module scope.
    # Imported (rather than duplicated) so the CLI defaults and the ones
    # run_acc_mismatch_diagnosis falls back to can never drift apart.
    from src.analysis.diagnose_acc import (
        DEFAULT_MODEL as DEFAULT_DIAG_MODEL,
        DEFAULT_DATASET as DEFAULT_DIAG_DATASET,
        DEFAULT_STAGE as DEFAULT_DIAG_STAGE,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved FP32 models instead of training from scratch"
    )
    parser.add_argument(
        "--deploy-int8-only",
        action="store_true",
        help="Load saved QAT models and run int8 conversion + accuracy gate only "
             "(no training, no Hessian/eigenvalue/SQNR analysis)"
    )
    parser.add_argument(
        "--benchmark-int8-only",
        action="store_true",
        help="Load saved FP32 baselines and deployed full-int8 models and run "
             "GPU latency/throughput benchmarking only (no training, no analysis)"
    )
    parser.add_argument(
        "--validate-pot-int8",
        action="store_true",
        help="Load saved PTQ/QAT models, reconstruct the deployed int8 models fresh, "
             "and run the per-layer PoT-preservation functional check only "
             "(no training, no Hessian/eigenvalue/SQNR analysis)"
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
        help="For --deploy-cpu-fbgemm and --diagnose-acc-mismatch: evaluate accuracy on only the "
             "first N batches of the validation set (useful for IMAGENET100 at 224x224 on CPU). "
             "Default: full validation set."
    )
    parser.add_argument(
        "--diagnose-acc-mismatch",
        action="store_true",
        help="Isolate why the same checkpoint evaluates to a different top-1 accuracy locally "
             "than on the cluster: fingerprints the checkpoints, the class-to-index mapping and "
             "the transform pipeline, then evaluates the plain FP32 baseline and inspects "
             "label/per-class behaviour. Writes results/<RUN_ID>/logs/acc_mismatch_diagnosis.txt "
             "for direct diffing against a cluster run of the same mode. Reuses --checkpoint-dir, "
             "--load-run-id and --eval-subset. Intended for a local workstation run "
             "(python -m src.main ...), no SLURM/torchrun required."
    )
    parser.add_argument(
        "--diag-model",
        type=str,
        default=DEFAULT_DIAG_MODEL,
        help=f"Model to scope --diagnose-acc-mismatch to (default: {DEFAULT_DIAG_MODEL})"
    )
    parser.add_argument(
        "--diag-dataset",
        type=str,
        default=DEFAULT_DIAG_DATASET,
        help=f"Dataset to scope --diagnose-acc-mismatch to (default: {DEFAULT_DIAG_DATASET})"
    )
    parser.add_argument(
        "--diag-stage",
        type=str,
        default=DEFAULT_DIAG_STAGE,
        choices=["PTQ", "QAT"],
        help=f"Stage to scope --diagnose-acc-mismatch to (default: {DEFAULT_DIAG_STAGE})"
    )
    return parser.parse_args()