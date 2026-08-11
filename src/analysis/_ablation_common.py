"""
_ablation_common.py -- shared weight-ablation infrastructure reused across
weight_ablation_canonical.py, weight_ablation_diagnose.py, spike_layer_cause.py,
quant_induced_trace.py, and relock_traces.py: robust checkpoint-path
resolution, the eval-loader builder, the weight-mask verifier, and the Part 0
path-equivalence gate.

Extracted from src/analysis/weight_ablation.py (P1, retired -- superseded by
weight_ablation_canonical.py) when that module was removed; these are the
only symbols any retained module imported from it.
"""

import os
import re
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.analysis.diagnose_activations import (
    _load_quant_model,
    _load_fp32_reference,
    _disable_activation_quant,
    _disable_weight_quant,
)
from src.utility.config import TEST_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

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
        logger.info(f"[AblationCommon] Resolved checkpoint ({token_desc}) -> {resolved}")
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
# Weight-mask verification: exactly the given set of layers has an active
# (non-Identity) weight_fake_quant, every other quantized layer's
# weight_fake_quant is Identity, and every act_fake_quant is Identity. One
# generic check covers the isolation case (one active layer), the
# leave-one-out case (all but one active), the weights-only-via-Identity
# case (all active), and the both-Identity case (none active).
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
            f"[AblationCommon] {label}: SOFT anchor check drifted -- fp32 {fp32_acc:.2f}% vs "
            f"known {anchor['fp32_acc']:.2f}% (diff {fp32_diff:.2f}pt), weights_only "
            f"{weights_only_acc:.2f}% vs known {anchor['weights_only_acc']:.2f}% "
            f"(diff {weights_diff:.2f}pt) -- WARNING only, not a hard gate failure (see the "
            f"path-equivalence gate for the hard check)."
        )
    else:
        logger.info(
            f"[AblationCommon] {label}: soft anchor check OK -- fp32 {fp32_acc:.2f}% "
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
    -- the one the isolation sweep's leave-one-out baseline and damage
    figures are defined against.
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
        f"[AblationCommon] {label}: fp32_acc={fp32_acc:.2f}% "
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
        f"[AblationCommon] {label}: path-equivalence gate -- "
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
            f"[AblationCommon] {label}: optional end-to-end check (PTQ) -- both-Identity "
            f"acc {acc_both_identity:.2f}% vs FP32 reference {fp32_acc:.2f}% "
            f"(diff {both_identity_diff:.2f}pt) -- informational only, not gated"
        )
    else:
        logger.info(
            f"[AblationCommon] {label}: optional end-to-end check (QAT) -- both-Identity "
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
