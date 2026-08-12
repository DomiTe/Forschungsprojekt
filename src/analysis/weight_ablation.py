"""
weight_ablation.py -- measures each layer's *weight*-quantization damage in
isolation and correlates it against that layer's precomputed weight-Hessian
trace.

Motivation: the PoT PTQ accuracy collapse is ~70pts activation-driven and
~5pts weight-driven (src/analysis/diagnose_activations.py). The weight-Hessian
story can therefore only be tested against weight-quant damage specifically.
Naively "quantizing only layer l" is invalid: a QuantizedConv2d/QuantizedLinear
carries both weight_fake_quant and act_fake_quant, so it would also turn on
that layer's activation quantizer -- for the high-outlier-factor layers the
measured damage would be activation-clipping damage the weight trace does not
predict. This mode isolates weights by disabling ALL activation quantization
first, then quantizes one layer's weights at a time.

Three parts, per model x dataset x stage (CIFAR10; resnet18_no_weights and
resnet50_no_weights required, cnn included as a bonus; PTQ, QAT):

  Part 0 (gate): compute FP32 reference accuracy, weights-only accuracy via
    the act_fake_quant->Identity swap, and weights-only accuracy via
    bake_pot_into_standard_layers (a structurally independent construction
    that also drops activation quantization). Both weights-only constructions
    quantize the same weights with activations untouched, so they must agree
    to within PATH_EQUIVALENCE_TOLERANCE_PTS -- this is a *self-consistent*
    hard gate (no remembered constant required) that catches a broken
    Identity-swap before it silently corrupts the isolation sweep. A
    secondary, non-fatal soft check compares against independently-known
    accuracies (currently only resnet50/CIFAR10/PTQ) purely as a sanity note.
  Part 1: reconstruct the model fresh per layer, isolate that one layer's
    weight quantization (everything else FP32, all activations FP32),
    evaluate on the full validation set. Optionally also runs the
    leave-one-out complement (every layer quantized except l) for direct
    comparison against the existing conv1 exclusion result
    (src/analysis/layer_ablation.py: excluding conv1 recovered 3.07pts).
  Part 2: Spearman correlation (+ top-5 overlap, tie-immune) between
    per-layer weight_damage_pts and the precomputed weight-Hessian trace,
    reusing (not recomputing) layerwise_hessian_traces.csv via
    src/analysis/layer_ablation.py's existing loader.

Analysis only -- no torchao, no INT8 conversion, no deployment/benchmark
path. Reuses (does not duplicate) the checkpoint loader, evaluation function,
bake_pot_into_standard_layers, and Identity-swap helpers already established
in src/analysis/diagnose_activations.py and src/main.py, and the Hessian
trace CSV loader in src/analysis/layer_ablation.py.

Checkpoint filenames in this project are not perfectly uniform (stray
spaces/dots/typos have appeared), so every checkpoint path here is resolved
by normalized-token matching (see _resolve_checkpoint_robust) rather than an
exact f-string, and the resolved absolute path is logged before loading. A
"missing checkpoint" skip is only ever issued for genuine absence -- a
near-miss (file present under a differently-formatted name) or an ambiguous
match (multiple candidates) raises loudly instead of silently skipping.

Runs as a single local process (`python -m src.main --weight-ablation ...`),
CPU or CUDA -- prefers CUDA when available, since this sweep reconstructs
and evaluates the model fresh per layer (resnet50: ~53 layers x up to 2
evaluations x 2 stages).
"""

import os
import re
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.quantization.deploy_fbgemm import MODELS, DATASETS, STAGES, _resolve_checkpoint_dir
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _load_quant_model,
    _load_fp32_reference,
    _disable_activation_quant,
    _disable_weight_quant,
    _append_row,
)
from src.analysis.layer_ablation import (
    _resolve_hessian_csv_path,
    _load_hessian_traces,
    _traces_for_combo,
)
from src.utility.config import CSV_DIR, DATASET_SPECS, TEST_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

SEED = None

# Hard gate (Part 0): the act_fake_quant->Identity construction and the
# bake_pot_into_standard_layers construction both apply the same weight
# quantizer to the same weights with activations untouched, so they must
# land within a fraction of a point of each other (a few samples' worth of
# noise at most). This is self-consistent per model x stage -- it needs no
# remembered "expected" accuracy table.
PATH_EQUIVALENCE_TOLERANCE_PTS = 0.1

# Soft sanity note only (Part 0): independently-known accuracies from prior
# runs, logged as a WARNING (not a hard failure) if the freshly-computed
# numbers drift far from them. Combos not listed here have no known
# reference yet and are simply not checked.
SOFT_KNOWN_ANCHORS = {
    ("resnet50_no_weights", "CIFAR10", "PTQ"): {"fp32_acc": 80.56, "weights_only_acc": 75.46},
}
SOFT_ANCHOR_TOLERANCE_PTS = 1.0

ABLATION_FIELDNAMES = [
    "model", "dataset", "stage", "layer", "hessian_trace", "fp32_acc", "isolated_acc",
    "weight_damage_pts", "leave_one_out_acc", "leave_one_out_recovery_pts",
]
CORRELATION_FIELDNAMES = [
    "model", "dataset", "stage", "n_layers_matched", "spearman_rho", "spearman_p",
    "top5_overlap", "note",
]


class WeightAblationError(RuntimeError):
    pass


class WeightAblationCheckpointError(RuntimeError):
    """
    Raised when checkpoint resolution finds either an ambiguous match (more
    than one candidate) or a near-miss (a file that matches all but one
    required token, suggesting it IS the intended checkpoint under a
    differently-formatted name). Deliberately a distinct type from
    FileNotFoundError -- callers must not treat this as a legitimate
    "checkpoint genuinely doesn't exist" skip.
    """
    pass


# ---------------------------------------------------------------------------
# Robust checkpoint path resolution
# ---------------------------------------------------------------------------

def _normalize_token(s: str) -> str:
    # Case-insensitive, treats spaces/dots/underscores/hyphens as equivalent
    # (stripped entirely) so "PTQ", "ptq", " ptq", "ptq." etc. all normalize
    # the same, and "resnet50_no_weights" / "resnet50.no.weights" /
    # "ResNet50 No Weights" all normalize to "resnet50noweights".
    return re.sub(r"[\s._-]+", "", s.lower())


def _resolve_checkpoint_robust(directory: str, tokens: dict[str, str]) -> str:
    """
    Finds the single .pt file in `directory` whose normalized filename
    contains every value in `tokens` as a substring, and returns its
    absolute path (logged). Distinguishes three failure modes:

    - genuinely absent: no file matches even len(tokens)-1 tokens -> raises
      FileNotFoundError (the only case a caller should treat as a legitimate
      graceful skip).
    - near-miss: no file matches all tokens, but some file matches all but
      one -> raises WeightAblationCheckpointError (the file is very likely
      present under a differently-formatted name; do not silently skip).
      Only attempted when there are >= 3 required tokens -- with only 2
      tokens (e.g. model+dataset for the FP32 baseline, no stage token),
      a 1-token partial match is too weak a signal to distinguish "this is
      the intended file with a typo" from "this is simply a different
      model's/dataset's file that happens to share one token" (every
      CIFAR10 baseline shares the dataset token, for instance).
    - ambiguous: more than one file matches all tokens -> raises
      WeightAblationCheckpointError (refuses to guess).
    """
    token_desc = ", ".join(f"{k}={v}" for k, v in tokens.items())

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory} (looking for {token_desc})")

    norm_tokens = {key: _normalize_token(val) for key, val in tokens.items()}
    all_files = sorted(f for f in os.listdir(directory) if f.endswith(".pt"))

    full_matches = [
        fname for fname in all_files
        if all(tok in _normalize_token(fname[:-len(".pt")]) for tok in norm_tokens.values())
    ]

    if len(full_matches) == 1:
        resolved = os.path.abspath(os.path.join(directory, full_matches[0]))
        logger.info(f"[WeightAblation] Resolved checkpoint ({token_desc}) -> {resolved}")
        return resolved

    if len(full_matches) > 1:
        resolved_list = [os.path.abspath(os.path.join(directory, f)) for f in full_matches]
        raise WeightAblationCheckpointError(
            f"ambiguous checkpoint match for {token_desc} in {directory}: {len(full_matches)} "
            f"files match all tokens: {resolved_list} -- refusing to guess which is correct."
        )

    if len(norm_tokens) >= 3:
        near_misses = [
            fname for fname in all_files
            if sum(1 for tok in norm_tokens.values() if tok in _normalize_token(fname[:-len(".pt")]))
            == len(norm_tokens) - 1
        ]
        if near_misses:
            raise WeightAblationCheckpointError(
                f"no exact token match for {token_desc} in {directory}, but found "
                f"{len(near_misses)} near-miss candidate(s) matching all but one token: "
                f"{near_misses} -- this is NOT a genuine 'missing checkpoint', the file is "
                f"likely present under a differently-formatted name. Refusing to silently "
                f"skip; fix the normalization or the filename."
            )

    raise FileNotFoundError(
        f"genuinely no checkpoint found for {token_desc} in {directory} "
        f"({len(all_files)} .pt files present: {all_files})"
    )


# ---------------------------------------------------------------------------
# Eval loader: num_workers=0, pin_memory=False, shuffle=False (explicit, per
# this mode's constraints -- overrides the get_data_loaders default of
# num_workers=8 / PIN_MEMORY=True on CUDA machines).
# ---------------------------------------------------------------------------

def _build_eval_loader(dataset_name: str) -> tuple[DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    eval_loader = DataLoader(
        val_loader.dataset, batch_size=TEST_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    return eval_loader, num_classes


# ---------------------------------------------------------------------------
# Weight-mask verification (Part 1 step 3): exactly the given set of layers
# has an active (non-Identity) weight_fake_quant, every other quantized
# layer's weight_fake_quant is Identity, and every act_fake_quant is
# Identity. One generic check covers the isolation case (one active layer),
# the leave-one-out case (all but one active), Part 0's weights-only-via-
# Identity case (all active), and Part 0's both-Identity case (none active).
# ---------------------------------------------------------------------------

def _verify_weight_mask(model: nn.Module, expected_active_layers: set[str], label: str) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if not isinstance(module.act_fake_quant, nn.Identity):
            raise WeightAblationError(
                f"{label}: '{name}.act_fake_quant' is not nn.Identity (got "
                f"{type(module.act_fake_quant).__name__}) -- activation quantization leaked "
                f"into a weight-isolation measurement, damage figure would be meaningless."
            )
        is_active = not isinstance(module.weight_fake_quant, nn.Identity)
        should_be_active = name in expected_active_layers
        if is_active != should_be_active:
            raise WeightAblationError(
                f"{label}: '{name}.weight_fake_quant' active={is_active}, expected "
                f"active={should_be_active} -- weight-mask verification failed, evaluating now "
                f"would silently measure the wrong configuration."
            )


# ---------------------------------------------------------------------------
# Part 0: anchors + self-consistent path-equivalence gate
# ---------------------------------------------------------------------------

def _soft_anchor_check(model_name: str, dataset_name: str, stage: str, fp32_acc: float, weights_only_acc: float) -> None:
    label = f"{stage} {model_name}/{dataset_name}"
    anchor = SOFT_KNOWN_ANCHORS.get((model_name, dataset_name, stage))
    if anchor is None:
        return

    fp32_diff = abs(fp32_acc - anchor["fp32_acc"])
    weights_diff = abs(weights_only_acc - anchor["weights_only_acc"])
    if fp32_diff > SOFT_ANCHOR_TOLERANCE_PTS or weights_diff > SOFT_ANCHOR_TOLERANCE_PTS:
        logger.warning(
            f"[WeightAblation] {label}: SOFT anchor check drifted -- fp32 {fp32_acc:.2f}% vs "
            f"known {anchor['fp32_acc']:.2f}% (diff {fp32_diff:.2f}pt), weights_only "
            f"{weights_only_acc:.2f}% vs known {anchor['weights_only_acc']:.2f}% "
            f"(diff {weights_diff:.2f}pt) -- WARNING only, not a hard gate failure (see the "
            f"path-equivalence gate for the hard check)."
        )
    else:
        logger.info(
            f"[WeightAblation] {label}: soft anchor check OK -- fp32 {fp32_acc:.2f}% "
            f"(known {anchor['fp32_acc']:.2f}%), weights_only {weights_only_acc:.2f}% "
            f"(known {anchor['weights_only_acc']:.2f}%)"
        )


def _run_part0(
    model_name: str, dataset_name: str, stage: str,
    quant_ckpt_path: str, fp32_ckpt_path: str,
    num_classes: int, channels: int, image_size: int,
    eval_loader: DataLoader, device: torch.device,
) -> tuple[bool, float, float, list[str], str | None]:
    """
    Returns (gate_passed, fp32_acc, weights_only_acc, all_layer_names, fail_note).
    weights_only_acc is the act_fake_quant->Identity construction's accuracy
    -- the one Part 1's leave-one-out baseline and damage figures are
    defined against.
    """
    from src.main import evaluate, bake_pot_into_standard_layers

    label = f"{stage} {model_name}/{dataset_name}"

    # 1. FP32 reference.
    fp32_model = _load_fp32_reference(model_name, fp32_ckpt_path, num_classes, channels, image_size).to(device)
    fp32_acc = evaluate(fp32_model, eval_loader, device)
    del fp32_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Weights-only via act_fake_quant -> Identity (the construction Part 1 depends on).
    model_identity, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
    model_identity = model_identity.to(device)
    all_layer_names = [
        name for name, module in model_identity.named_modules()
        if isinstance(module, (QuantizedConv2d, QuantizedLinear))
    ]
    _disable_activation_quant(model_identity)
    _verify_weight_mask(model_identity, set(all_layer_names), f"{label} weights-only(identity)")
    acc_identity = evaluate(model_identity, eval_loader, device)
    del model_identity
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Weights-only via baking (structurally independent trusted reference).
    model_to_bake, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
    model_to_bake = model_to_bake.to(device)
    baked_model = bake_pot_into_standard_layers(model_to_bake).to(device)
    acc_baked = evaluate(baked_model, eval_loader, device)
    del model_to_bake, baked_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        f"[WeightAblation] {label}: fp32_acc={fp32_acc:.2f}% "
        f"weights_only(act->Identity)={acc_identity:.2f}% weights_only(baked)={acc_baked:.2f}% "
        f"({len(all_layer_names)} layers)"
    )

    # Hard gate: the two weights-only constructions must agree -- they apply
    # the identical weight quantizer to identical weights with activations
    # untouched, so any divergence beyond a few samples means the Identity
    # swap the isolation sweep depends on is not equivalent to baking.
    path_diff = abs(acc_identity - acc_baked)
    gate_passed = path_diff < PATH_EQUIVALENCE_TOLERANCE_PTS
    logger.info(
        f"[WeightAblation] {label}: path-equivalence gate -- "
        f"|{acc_identity:.2f}% - {acc_baked:.2f}%| = {path_diff:.3f}pt "
        f"(tolerance < {PATH_EQUIVALENCE_TOLERANCE_PTS}pt) -- {'PASS' if gate_passed else 'FAIL'}"
    )

    _soft_anchor_check(model_name, dataset_name, stage, fp32_acc, acc_identity)

    # Optional end-to-end check: both quantizers -> Identity should reproduce
    # the FP32 reference for PTQ (weights untouched by PTQ training, fusion
    # is exact in eval). For QAT this instead reflects the QAT-trained
    # weights' own clean-forward accuracy -- QAT continued training beyond
    # the frozen FP32 baseline checkpoint, so it is NOT expected to match
    # fp32_acc; logged purely as an informational note, never gated.
    model_both, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
    model_both = model_both.to(device)
    _disable_activation_quant(model_both)
    _disable_weight_quant(model_both)
    _verify_weight_mask(model_both, set(), f"{label} both-identity")
    acc_both_identity = evaluate(model_both, eval_loader, device)
    del model_both
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if stage == "PTQ":
        both_identity_diff = abs(acc_both_identity - fp32_acc)
        logger.info(
            f"[WeightAblation] {label}: optional end-to-end check (PTQ) -- both-Identity "
            f"acc {acc_both_identity:.2f}% vs FP32 reference {fp32_acc:.2f}% "
            f"(diff {both_identity_diff:.2f}pt) -- informational only, not gated"
        )
    else:
        logger.info(
            f"[WeightAblation] {label}: optional end-to-end check (QAT) -- both-Identity "
            f"acc {acc_both_identity:.2f}% (informational: reflects the QAT-trained weights' "
            f"own clean-forward accuracy, NOT expected to match the frozen FP32 baseline "
            f"{fp32_acc:.2f}% since QAT continued training beyond that checkpoint)"
        )

    if not gate_passed:
        note = (
            f"path-equivalence gate failed: |act->Identity {acc_identity:.2f}% - "
            f"baked {acc_baked:.2f}%| = {path_diff:.3f}pt >= {PATH_EQUIVALENCE_TOLERANCE_PTS}pt "
            f"-- the act->Identity swap is not equivalent to baking, the isolation sweep would "
            f"be unsound"
        )
        return False, fp32_acc, acc_identity, all_layer_names, note

    return True, fp32_acc, acc_identity, all_layer_names, None


# ---------------------------------------------------------------------------
# Part 1: per-layer weights-only isolation (+ optional leave-one-out)
# ---------------------------------------------------------------------------

def _run_isolation_sweep(
    model_name: str, dataset_name: str, stage: str, quant_ckpt_path: str,
    num_classes: int, channels: int, image_size: int,
    eval_loader: DataLoader, device: torch.device,
    fp32_acc: float, weights_only_all_acc: float, all_layer_names: list[str],
    hessian_traces: dict[str, float], output_csv_path: str,
    run_leave_one_out: bool = True,
) -> list[dict]:
    from src.main import evaluate

    label = f"{stage} {model_name}/{dataset_name}"
    rows: list[dict] = []

    for layer_name in all_layer_names:
        other_layers = {n for n in all_layer_names if n != layer_name}

        model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
        model = model.to(device)
        _disable_activation_quant(model)
        _disable_weight_quant(model, layer_names=other_layers)
        _verify_weight_mask(model, {layer_name}, f"{label} isolate={layer_name}")
        isolated_acc = evaluate(model, eval_loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_damage_pts = fp32_acc - isolated_acc

        row = {
            "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer_name,
            "hessian_trace": hessian_traces.get(layer_name, ""),
            "fp32_acc": fp32_acc, "isolated_acc": isolated_acc, "weight_damage_pts": weight_damage_pts,
            "leave_one_out_acc": "", "leave_one_out_recovery_pts": "",
        }

        if run_leave_one_out:
            loo_model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
            loo_model = loo_model.to(device)
            _disable_activation_quant(loo_model)
            _disable_weight_quant(loo_model, layer_names={layer_name})
            _verify_weight_mask(loo_model, other_layers, f"{label} leave-one-out={layer_name}")
            loo_acc = evaluate(loo_model, eval_loader, device)
            del loo_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row["leave_one_out_acc"] = loo_acc
            row["leave_one_out_recovery_pts"] = loo_acc - weights_only_all_acc

        loo_msg = (
            f" leave_one_out_recovery={row['leave_one_out_recovery_pts']:.2f}pts"
            if run_leave_one_out else ""
        )
        logger.info(
            f"[WeightAblation] {label} layer={layer_name}: isolated_acc={isolated_acc:.2f}% "
            f"weight_damage={weight_damage_pts:.2f}pts{loo_msg}"
        )

        _append_row(output_csv_path, row, ABLATION_FIELDNAMES)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Part 2: correlation with the precomputed weight-Hessian trace
# ---------------------------------------------------------------------------

def _run_correlation(
    model_name: str, dataset_name: str, stage: str,
    ablation_rows: list[dict], hessian_traces: dict[str, float], output_csv_path: str,
) -> None:
    label = f"{stage} {model_name}/{dataset_name}"

    ablation_layer_names = {r["layer"] for r in ablation_rows}
    trace_layer_names = set(hessian_traces.keys())

    only_in_trace = sorted(trace_layer_names - ablation_layer_names)
    only_in_ablation = sorted(ablation_layer_names - trace_layer_names)
    if only_in_trace:
        logger.warning(
            f"[WeightAblation] {label}: layers in Hessian trace but not in ablation set "
            f"(excluded from correlation): {only_in_trace}"
        )
    if only_in_ablation:
        logger.warning(
            f"[WeightAblation] {label}: layers in ablation set but not in Hessian trace "
            f"(excluded from correlation): {only_in_ablation}"
        )

    matched_layers = [r["layer"] for r in ablation_rows if r["layer"] in hessian_traces]
    n_matched = len(matched_layers)

    if n_matched == 0:
        note = "no matched layers between ablation set and Hessian trace set -- correlation undefined"
        logger.error(f"[WeightAblation] {label}: {note}")
        _append_row(output_csv_path, {
            "model": model_name, "dataset": dataset_name, "stage": stage,
            "n_layers_matched": 0, "spearman_rho": "", "spearman_p": "",
            "top5_overlap": "", "note": note,
        }, CORRELATION_FIELDNAMES)
        return

    damage_by_layer = {r["layer"]: r["weight_damage_pts"] for r in ablation_rows}
    traces = [hessian_traces[l] for l in matched_layers]
    damages = [damage_by_layer[l] for l in matched_layers]

    if n_matched >= 3:
        rho, p_value = spearmanr(traces, damages)
    else:
        rho, p_value = float("nan"), float("nan")
        logger.warning(
            f"[WeightAblation] {label}: only {n_matched} matched layers -- Spearman correlation "
            f"not meaningful (need >= 3), reporting NaN"
        )

    k = min(5, n_matched)
    top_k_by_trace = set(sorted(matched_layers, key=lambda l: hessian_traces[l], reverse=True)[:k])
    top_k_by_damage = set(sorted(matched_layers, key=lambda l: damage_by_layer[l], reverse=True)[:k])
    overlap = len(top_k_by_trace & top_k_by_damage)
    top5_overlap = f"{overlap}/{k}"

    logger.info(
        f"[WeightAblation] {label}: n_matched={n_matched} spearman_rho={rho:.4f} "
        f"p={p_value:.4g} top{k}_overlap={top5_overlap}"
    )

    note = ""
    if only_in_trace or only_in_ablation:
        note = f"excluded {len(only_in_trace)} trace-only + {len(only_in_ablation)} ablation-only layers"

    _append_row(output_csv_path, {
        "model": model_name, "dataset": dataset_name, "stage": stage,
        "n_layers_matched": n_matched, "spearman_rho": rho, "spearman_p": p_value,
        "top5_overlap": top5_overlap, "note": note,
    }, CORRELATION_FIELDNAMES)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_weight_ablation(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    run_leave_one_out: bool = True,
) -> None:
    # torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[WeightAblation] CUDA not available -- falling back to CPU, this will be slow.")
    logger.info(f"[WeightAblation] device={device} seed={SEED}")

    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)
    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    hessian_csv_path = _resolve_hessian_csv_path(checkpoint_dir, load_run_id)
    hessian_df = _load_hessian_traces(hessian_csv_path)

    os.makedirs(CSV_DIR, exist_ok=True)
    ablation_csv = os.path.join(CSV_DIR, "weight_ablation.csv")
    correlation_csv = os.path.join(CSV_DIR, "weight_ablation_correlation.csv")

    for dataset_name in DATASETS:
        specs = DATASET_SPECS[dataset_name]
        channels, image_size = specs["channels"], specs["image_size"]

        try:
            eval_loader, num_classes = _build_eval_loader(dataset_name)
        except Exception as exc:
            logger.warning(f"[WeightAblation] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in MODELS:
            for stage in STAGES:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[WeightAblation] === {label} ===")

                try:
                    quant_ckpt_path = _resolve_checkpoint_robust(
                        resolved_checkpoint_dir,
                        {"stage": stage, "model": model_name, "dataset": dataset_name},
                    )
                    fp32_ckpt_path = _resolve_checkpoint_robust(
                        fp32_models_dir,
                        {"model": model_name, "dataset": dataset_name},
                    )
                except FileNotFoundError as exc:
                    logger.warning(f"[WeightAblation] {label}: checkpoint genuinely missing ({exc}) -- skipping")
                    continue
                except WeightAblationCheckpointError as exc:
                    logger.error(
                        f"[WeightAblation] {label}: checkpoint resolution AMBIGUOUS or NEAR-MISS, "
                        f"NOT a legitimate missing-checkpoint skip ({exc}) -- skipping this combo, "
                        f"but this needs human attention."
                    )
                    continue

                # ---- Part 0: anchors + path-equivalence gate ----
                gate_passed, fp32_acc, weights_only_all_acc, all_layer_names, note = _run_part0(
                    model_name, dataset_name, stage, quant_ckpt_path, fp32_ckpt_path,
                    num_classes, channels, image_size, eval_loader, device,
                )
                if not gate_passed:
                    logger.error(f"[WeightAblation] {label}: GATE FAILED -- {note}. Skipping Parts 1-2.")
                    continue

                # ---- Part 1: per-layer isolation ----
                combo_traces = _traces_for_combo(hessian_df, model_name, dataset_name, stage)
                ablation_rows = _run_isolation_sweep(
                    model_name, dataset_name, stage, quant_ckpt_path, num_classes, channels, image_size,
                    eval_loader, device, fp32_acc, weights_only_all_acc, all_layer_names, combo_traces,
                    ablation_csv, run_leave_one_out=run_leave_one_out,
                )

                # ---- Part 2: correlation with the precomputed weight-Hessian trace ----
                _run_correlation(model_name, dataset_name, stage, ablation_rows, combo_traces, correlation_csv)

    logger.info("[WeightAblation] === Weight-Ablation complete ===")
