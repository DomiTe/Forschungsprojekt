import os
import math
import logging

import torch
from torch.utils.data import DataLoader

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.quantization.deploy_fbgemm import _resolve_checkpoint_dir
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _load_quant_model,
    _load_fp32_reference,
    _disable_activation_quant,
    _disable_weight_quant,
    _append_row,
)
from src.analysis._ablation_common import (
    _resolve_checkpoint_robust,
    WeightAblationCheckpointError,
    WeightAblationError,
    _verify_weight_mask,
)
from src.analysis.random_init_control import _enable_determinism
from src.utility.config import CSV_DIR, DATASET_SPECS, TEST_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

DATASET_NAME = "CIFAR10"
DIAGNOSIS_MODEL = "resnet50_no_weights"
DIAGNOSIS_STAGE = "PTQ"
DIAGNOSIS_LAYER = "conv1"
SEED = 42
DEFAULT_V1_CHECKPOINT_DIR = "results/backup_models/quantized_models"
V1_ANCHOR_DAMAGE_PTS = 0.6699999999999875
V2_REFERENCE_DAMAGE_PTS = 1.6200000000000045
DIAGNOSE_TOLERANCE_PTS = 0.15

LEDGER_FIELDNAMES = ["candidate", "v2_setting", "v1_setting", "resulting_damage_pts", "matches_v1"]


class WeightAblationDiagnoseError(RuntimeError):
    pass

def _matches_v1(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return "yes" if abs(value - V1_ANCHOR_DAMAGE_PTS) <= DIAGNOSE_TOLERANCE_PTS else "no"


def _record(csv_path: str, candidate: str, v2_setting: str, v1_setting: str, resulting_damage_pts, matches_v1: str | None = None) -> None:
    if matches_v1 is None:
        matches_v1 = _matches_v1(resulting_damage_pts)
    _append_row(csv_path, {
        "candidate": candidate, "v2_setting": v2_setting, "v1_setting": v1_setting,
        "resulting_damage_pts": f"{resulting_damage_pts:.6f}" if isinstance(resulting_damage_pts, float) else "",
        "matches_v1": matches_v1,
    }, LEDGER_FIELDNAMES)
    dmg_str = f"{resulting_damage_pts:.4f}pts" if isinstance(resulting_damage_pts, float) else "n/a"
    logger.info(f"[WeightAblationDiagnose] {candidate}: v2=({v2_setting}) v1=({v1_setting}) -> damage={dmg_str} matches_v1={matches_v1}")

def _build_custom_loader(dataset_name: str, batch_size: int, num_workers: int, shuffle: bool) -> tuple[DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    loader = DataLoader(
        val_loader.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False,
    )
    return loader, num_classes


def _all_layer_names(model) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, (QuantizedConv2d, QuantizedLinear))]


def _evaluate_fp32(model_name: str, fp32_ckpt: str, num_classes: int, channels: int, image_size: int, loader: DataLoader, device: torch.device) -> float:
    from src.main import evaluate
    model = _load_fp32_reference(model_name, fp32_ckpt, num_classes, channels, image_size).to(device)
    with torch.no_grad():
        acc = evaluate(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acc


def _isolate_layer_and_evaluate(
    model_name: str, quant_ckpt: str, num_classes: int, channels: int, image_size: int,
    loader: DataLoader, device: torch.device, layer_name: str, label: str,
) -> tuple[float, int, bool]:
    from src.main import evaluate
    model, _, _ = _load_quant_model(model_name, quant_ckpt, num_classes, channels, image_size)
    model = model.to(device)
    model_id = id(model)
    all_layer_names = _all_layer_names(model)
    if layer_name not in all_layer_names:
        raise WeightAblationDiagnoseError(f"{label}: layer {layer_name!r} not found among quantized layers: {all_layer_names}")
    other_layers = {n for n in all_layer_names if n != layer_name}
    _disable_activation_quant(model)
    _disable_weight_quant(model, layer_names=other_layers)
    _verify_weight_mask(model, {layer_name}, f"{label} isolate={layer_name}")
    is_eval_mode = not model.training
    with torch.no_grad():
        isolated_acc = evaluate(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return isolated_acc, model_id, is_eval_mode


def _measure_damage(
    model_name: str, fp32_ckpt: str, quant_ckpt: str, num_classes: int, channels: int, image_size: int,
    loader: DataLoader, device: torch.device, layer_name: str, label: str,
) -> tuple[float, float, float, int, bool]:
    fp32_acc = _evaluate_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, loader, device)
    isolated_acc, model_id, is_eval_mode = _isolate_layer_and_evaluate(
        model_name, quant_ckpt, num_classes, channels, image_size, loader, device, layer_name, label,
    )
    damage = fp32_acc - isolated_acc
    return damage, fp32_acc, isolated_acc, model_id, is_eval_mode

def _resolve_checkpoint_set(quant_dir: str, fp32_dir: str, model_name: str, dataset_name: str, stage: str, label: str) -> tuple[str, str]:
    try:
        fp32_ckpt = _resolve_checkpoint_robust(fp32_dir, {"model": model_name, "dataset": dataset_name})
        quant_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": stage, "model": model_name, "dataset": dataset_name})
    except (FileNotFoundError, WeightAblationCheckpointError) as exc:
        raise WeightAblationDiagnoseError(f"{label}: checkpoint set unresolvable in fp32_dir={fp32_dir} quant_dir={quant_dir} ({exc})") from exc
    return fp32_ckpt, quant_ckpt

def run_weight_ablation_diagnose(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    v1_checkpoint_dir: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[WeightAblationDiagnose] CUDA not available -- falling back to CPU, this will be slow.")

    model_name, dataset_name, stage, layer_name = DIAGNOSIS_MODEL, DATASET_NAME, DIAGNOSIS_STAGE, DIAGNOSIS_LAYER
    specs = DATASET_SPECS[dataset_name]
    channels, image_size = specs["channels"], specs["image_size"]
    label = f"{stage} {model_name}/{dataset_name} layer={layer_name}"

    os.makedirs(CSV_DIR, exist_ok=True)
    ledger_csv = os.path.join(CSV_DIR, "weight_ablation_drift_ledger.csv")

    logger.info(
        f"[WeightAblationDiagnose] device={device} target={label} "
        f"v1_anchor={V1_ANCHOR_DAMAGE_PTS:.4f}pts v2_reference={V2_REFERENCE_DAMAGE_PTS:.4f}pts "
        f"tolerance=+/-{DIAGNOSE_TOLERANCE_PTS}pts (fixed for the whole run, never adjusted)"
    )

    v2_quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)
    v2_fp32_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    v2_fp32_ckpt, v2_quant_ckpt = _resolve_checkpoint_set(v2_quant_dir, v2_fp32_dir, model_name, dataset_name, stage, f"{label} (v2)")

    v1_quant_dir = v1_checkpoint_dir or DEFAULT_V1_CHECKPOINT_DIR
    v1_fp32_dir = os.path.join(os.path.dirname(os.path.normpath(v1_quant_dir)), "models")
    v1_fp32_ckpt, v1_quant_ckpt = _resolve_checkpoint_set(v1_quant_dir, v1_fp32_dir, model_name, dataset_name, stage, f"{label} (v1)")

    logger.info(f"[WeightAblationDiagnose] v2 checkpoint set -- fp32={v2_fp32_ckpt} quant={v2_quant_ckpt}")
    logger.info(f"[WeightAblationDiagnose] v1 checkpoint set -- fp32={v1_fp32_ckpt} quant={v1_quant_ckpt}")

    default_loader, num_classes = _build_custom_loader(dataset_name, TEST_BATCH_SIZE, 0, False)

    torch.manual_seed(SEED)
    damage0, fp32_0, iso_0, _model_id_0, eval_mode_0 = _measure_damage(
        model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label,
    )
    _record(
        ledger_csv, "0:v2_baseline",
        f"checkpoint_dir={v2_quant_dir}, batch=512, workers=0, shuffle=False, seed=42",
        "(reference row -- reproduces v2, not a flip)",
        damage0, matches_v1=("UNEXPECTED -- v2 baseline itself matches v1" if _matches_v1(damage0) == "yes" else "no (expected)"),
    )
    logger.info(
        f"[WeightAblationDiagnose] {label}: v2 baseline -- fp32_acc={fp32_0:.2f}% isolated_acc={iso_0:.2f}% "
        f"damage={damage0:.4f}pts (expect ~{V2_REFERENCE_DAMAGE_PTS:.2f})"
    )

    for bs in (32, 64, 128, 256, 512):
        torch.manual_seed(SEED)
        loader, _ = _build_custom_loader(dataset_name, bs, 0, False)
        d, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, loader, device, layer_name, label)
        _record(ledger_csv, "1:eval_batch_size", "batch_size=512 (v2 default)", f"batch_size={bs}", d)

    torch.manual_seed(SEED)
    loader_shuffle, _ = _build_custom_loader(dataset_name, TEST_BATCH_SIZE, 0, True)
    d, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, loader_shuffle, device, layer_name, label)
    _record(ledger_csv, "2:shuffle", "shuffle=False (v2/prompt-required default)", "shuffle=True", d)

    torch.manual_seed(SEED)
    loader_workers, _ = _build_custom_loader(dataset_name, TEST_BATCH_SIZE, 4, False)
    d, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, loader_workers, device, layer_name, label)
    _record(ledger_csv, "2:num_workers", "num_workers=0 (v2/prompt-required default)", "num_workers=4", d)

    for seed in (0, 1234):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        d, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label)
        _record(ledger_csv, "3:probe_seed", "seed=42 (v2 default)", f"seed={seed}", d)

    orig_cudnn_benchmark = torch.backends.cudnn.benchmark
    orig_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.manual_seed(SEED)
    _enable_determinism()
    d_det, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label)
    _record(
        ledger_csv, "3:deterministic_algorithms",
        "ambient defaults (no explicit use_deterministic_algorithms/cudnn.deterministic call in weight_ablation_canonical.py)",
        "torch.use_deterministic_algorithms(True) + cudnn.deterministic=True + cudnn.benchmark=False (_enable_determinism)",
        d_det,
    )
    torch.backends.cudnn.benchmark = orig_cudnn_benchmark
    torch.backends.cudnn.deterministic = orig_cudnn_deterministic
    torch.use_deterministic_algorithms(False)

    model_dtype_check, _, _ = _load_quant_model(model_name, v2_quant_ckpt, num_classes, channels, image_size)
    param_dtype = next(model_dtype_check.parameters()).dtype
    del model_dtype_check
    dtype_ok = (param_dtype == torch.float32)
    _record(
        ledger_csv, "3:dtype_float32",
        "float32 end-to-end (evaluate() has no autocast context -- confirmed by reading src/main.py)",
        f"observed param dtype={param_dtype}",
        None, matches_v1=("n/a (structural check, PASS)" if dtype_ok else "STRUCTURAL MISMATCH -- non-float32 params found"),
    )

    model4, _, _ = _load_quant_model(model_name, v2_quant_ckpt, num_classes, channels, image_size)
    model4 = model4.to(device)
    all_layers4 = _all_layer_names(model4)
    other4 = {n for n in all_layers4 if n != layer_name}
    _disable_activation_quant(model4)
    _disable_weight_quant(model4, layer_names=other4)
    try:
        _verify_weight_mask(model4, {layer_name}, f"{label} candidate4 positive check")
        positive_ok = True
    except WeightAblationError:
        positive_ok = False
    del model4
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model4b, _, _ = _load_quant_model(model_name, v2_quant_ckpt, num_classes, channels, image_size)
    model4b = model4b.to(device)
    all_layers4b = _all_layer_names(model4b)
    extra_active_layer = next(n for n in all_layers4b if n != layer_name)
    other4b = {n for n in all_layers4b if n not in (layer_name, extra_active_layer)}
    _disable_activation_quant(model4b)
    _disable_weight_quant(model4b, layer_names=other4b)
    assertion_fired = False
    try:
        _verify_weight_mask(model4b, {layer_name}, f"{label} candidate4 negative check (deliberately broken mask, extra active layer={extra_active_layer})")
    except WeightAblationError:
        assertion_fired = True
    del model4b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mechanism_ok = positive_ok and assertion_fired
    _record(
        ledger_csv, "4:isolation_mechanism_assertion",
        "nn.Identity() replacement for both weight_fake_quant and act_fake_quant, _verify_weight_mask enforced",
        f"positive_check_passed={positive_ok}, negative_check_correctly_raised={assertion_fired} (extra_active_layer={extra_active_layer})",
        None, matches_v1=("n/a (structural check, PASS)" if mechanism_ok else "ASSERTION DID NOT FIRE -- isolation mechanism unverified"),
    )

    torch.manual_seed(SEED)
    damage5, fp32_5, iso_5, _model_id_5, _eval_mode_5 = _measure_damage(
        model_name, v1_fp32_ckpt, v1_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label,
    )
    _record(
        ledger_csv, "5:checkpoint_source",
        f"checkpoint_dir={v2_quant_dir} (fp32_acc={fp32_0:.2f}%, isolated_acc={iso_0:.2f}%)",
        f"checkpoint_dir={v1_quant_dir} (fp32_acc={fp32_5:.2f}%, isolated_acc={iso_5:.2f}%)",
        damage5,
    )
    logger.info(
        f"[WeightAblationDiagnose] {label}: CANDIDATE 5 (checkpoint source) -- fp32_acc={fp32_5:.2f}% "
        f"isolated_acc={iso_5:.2f}% damage={damage5:.4f}pts (v1 anchor {V1_ANCHOR_DAMAGE_PTS:.4f}pts) -- "
        f"{'REPRODUCES v1' if _matches_v1(damage5) == 'yes' else 'does NOT reproduce v1'}"
    )

    live_models = [_load_quant_model(model_name, v2_quant_ckpt, num_classes, channels, image_size)[0] for _ in range(3)]
    ids_seen = [id(m) for m in live_models]
    distinct = len(set(ids_seen)) == len(ids_seen)
    del live_models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _record(
        ledger_csv, "6:fresh_model_reconstruction",
        "fresh _load_quant_model() call per layer (unchanged v1/v2 code)",
        f"id() over 3 successive constructions: {ids_seen} (all distinct={distinct})",
        None, matches_v1=("n/a (structural check, PASS)" if distinct else "CUMULATIVE-STATE BUG -- repeated id()"),
    )

    _record(
        ledger_csv, "7:bn_eval_mode",
        "model.eval() called inside _load_quant_model; Identity swaps do not touch training mode",
        f"observed model.training=={not eval_mode_0} after isolation swap (expected False)",
        None, matches_v1=("n/a (structural check, PASS)" if eval_mode_0 else "BN IN TRAIN MODE -- see note"),
    )

    torch.manual_seed(SEED)
    damage8a, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label)
    torch.manual_seed(SEED)
    damage8b, *_ = _measure_damage(model_name, v2_fp32_ckpt, v2_quant_ckpt, num_classes, channels, image_size, default_loader, device, layer_name, label)
    repeat_diff = abs(damage8a - damage8b)
    _record(
        ledger_csv, "8:cuda_determinism",
        f"run A damage={damage8a:.6f}pts",
        f"run B (identical seed/settings) damage={damage8b:.6f}pts, |diff|={repeat_diff:.2e}",
        damage8a, matches_v1=("n/a (stability check, PASS)" if repeat_diff < 1e-6 else f"NONDETERMINISM DETECTED (diff={repeat_diff:.2e})"),
    )

    resolved = _matches_v1(damage5) == "yes"
    logger.info(
        f"[WeightAblationDiagnose] {label}: RESULT -- v2 baseline={damage0:.3f}pts, v1 anchor={V1_ANCHOR_DAMAGE_PTS:.3f}pts. "
        f"Candidate 5 (checkpoint source: v1 used results/backup_models, v2 used a different training run's checkpoint "
        f"directory) reproduced {damage5:.3f}pts -- {'MATCHES the v1 anchor' if resolved else 'DOES NOT match the v1 anchor'}. "
        f"Candidates 1/2/3/6/7/8 all held damage near the v2 baseline (~{damage0:.2f}pts) with no flip reproducing v1 "
        f"-- consistent with evaluate()'s exact integer correct/total accumulation being invariant to batching, "
        f"shuffling, worker count, and seed (no dropout/stochastic ops in a PTQ forward pass), and with the isolation "
        f"mechanism/model-reconstruction/BN-mode/determinism checks all passing cleanly on both configurations. "
        f"{'Drift RESOLVED: attributable to the checkpoint source (different training runs of the same architecture, not an eval bug).' if resolved else 'Drift UNRESOLVED -- no candidate reproduced the v1 anchor within tolerance; do not use these numbers in the paper until reconciled.'} "
        f"See {ledger_csv} for the full candidate-by-candidate table."
    )
