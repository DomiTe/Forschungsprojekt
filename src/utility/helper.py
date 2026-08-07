import argparse

def parse_args():
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
    return parser.parse_args()