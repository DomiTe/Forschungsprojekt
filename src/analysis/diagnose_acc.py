"""
diagnose_acc.py -- isolates why the *same* checkpoint evaluates to 7.64%
top-1 on the local CPU run but 29.10% on the cluster GPU run
(cnn / IMAGENET100 / PTQ).

The INT8-vs-FP32 delta is fine on both sides (-0.24%), so quantization is
not the cause: the FP32 baseline itself must already differ. The local
IMAGENET100 loader also reports exactly the cluster's dimensions
(130000 train / 5000 val across 100 classes), which rules out a truncated
dataset. That leaves three hypotheses:

  H1  class-to-index label mapping differs between the two filesystems
      (ImageFolder derives labels from directory order -- if anything in
      that chain is unsorted, the same image gets a different integer
      label on a different machine, and accuracy collapses to roughly
      chance while every dimension still looks correct).
  H2  the validation transform pipeline differs from what training used
      (most likely a missing/misapplied Normalize -- the network then sees
      inputs in [0,1] instead of standardized ImageNet statistics).
  H3  the local checkpoints are simply from a different training run than
      the numbers reported by the cluster.

Method: run all seven checks below unconditionally, in order, and write
everything to results/<RUN_ID>/logs/acc_mismatch_diagnosis.txt in addition
to the logger, so a local report and a cluster report of the same mode can
be diffed line by line.

  1. checkpoint identity  (H3)  file hash/size/mtime + per-tensor stats
  2. class mapping        (H1)  class list, ordering, hash, derivation
  3. transform pipeline   (H2)  full repr, crop dims, normalization
  4. input statistics     (H2)  what the network actually receives
  5. FP32 baseline eval   (the decisive split: data/preprocessing vs.
                           downstream PoT baking / quantized loading)
  6. label sanity               predictions vs. truth, offset scan
  7. per-class spread           permutation-flat vs. genuinely weak

Deliberately reuses the *existing* loader (src.utility.utils.get_data_loaders)
and the existing builder (src.model_cnn.train.build_model). The point is to
diagnose the path the pipeline really takes, not to bypass it with a
hand-rolled dataset that would prove nothing about the discrepancy.

Runs as a single local process (`python -m src.main --diagnose-acc-mismatch
...`), no torchrun/SLURM/torch.distributed involved -- but it is written to
run unchanged on the cluster GPU too, which is the whole point of the
report file.
"""

import os
import json
import hashlib
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utility.config import RESULTS_DIR, RUN_ID, LOG_DIR, DATASET_SPECS
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

# Default scope. Overridable from the CLI via --diag-model/--diag-dataset/
# --diag-stage; this is the combination the mismatch was observed on.
DEFAULT_MODEL = "cnn"
DEFAULT_DATASET = "IMAGENET100"
DEFAULT_STAGE = "PTQ"

REPORT_FILENAME = "acc_mismatch_diagnosis.txt"

# ---------------------------------------------------------------------------
# The numbers this mode exists to explain, kept here so the report can state
# a verdict instead of leaving raw figures to be interpreted by hand.
# The FP32 reference is derived, not measured: the cluster's INT8 run landed
# at 29.10% with an INT8-vs-FP32 delta of -0.24%, so its FP32 baseline was
# ~29.34%. That is what check 5 is compared against.
# ---------------------------------------------------------------------------
LOCAL_INT8_ACC = 7.64
CLUSTER_INT8_ACC = 29.10
CLUSTER_INT8_VS_FP32_DELTA = -0.24
EXPECTED_CLUSTER_FP32_ACC = CLUSTER_INT8_ACC - CLUSTER_INT8_VS_FP32_DELTA

# check 5: how close the local FP32 baseline has to land to count as "matches
# the cluster" -- generous, because the verdict only needs to separate
# "same ballpark" from "collapsed to near chance".
FP32_MATCH_TOLERANCE_PCT = 5.0

# check 7: if not one single class clears this, no class works at all, which
# is the signature of a label permutation rather than a weak model.
UNIFORM_LOW_BEST_CLASS_PCT = 20.0

# check 6: how far to scan for a systematic index offset between predicted
# and true labels (pred + k) % num_classes.
OFFSET_SCAN_RANGE = 5

CLASS_HEAD_COUNT = 15   # first N entries of class_to_idx to print
CLASS_TAIL_COUNT = 5    # last N entries of class_to_idx to print
SAMPLE_ROWS = 10        # pred/true rows to print side by side
TOP_K_LABELS = 5        # most-frequent predicted/true indices to print
BEST_WORST_CLASSES = 5  # best/worst per-class accuracies to print
PROBE_TENSOR_COUNT = 3  # named weight tensors to fingerprint per checkpoint

# Fingerprinted first when present, so the local and cluster reports probe
# the same tensor. Everything else is picked deterministically (see
# _select_probe_tensors), which keeps the two reports comparable per model.
PREFERRED_PROBE_TENSORS = ("conv1.weight",)

# Full-size TEST_BATCH_SIZE (256) at 224x224 is ~154MB of float32 per batch
# on CPU; pinning that previously contributed to an OOM. Batch size has no
# effect on top-1 accuracy, so the diagnostic loader caps it.
DIAG_MAX_BATCH_SIZE = 64

# Manifest files that, if present next to the data, could in principle carry
# an authoritative class order -- reported so it is visible whether the
# loader ignores one that exists.
MANIFEST_CANDIDATES = ("Labels.json", "labels.json", "classes.txt", "wnids.txt")


# ---------------------------------------------------------------------------
# Report sink: everything goes to the logger AND to the report file
# ---------------------------------------------------------------------------

class _Report:
    """
    Every diagnostic line is emitted twice: through the logger (so it lands
    in experiment_log.txt alongside the rest of the run) and into a buffer
    written to its own file at the end. The file exists specifically so a
    local run and a cluster run of this mode can be diffed directly, which
    is why the logger's timestamp/level prefix is not part of the buffered
    text.
    """

    def __init__(self, prefix: str = "[DiagAcc]"):
        self._lines: list[str] = []
        self._prefix = prefix

    def line(self, text: str = "") -> None:
        self._lines.append(text)
        logger.info(f"{self._prefix} {text}" if text else self._prefix)

    def section(self, title: str) -> None:
        self.line("")
        self.line("=" * 78)
        self.line(title)
        self.line("=" * 78)

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")
        logger.info(f"{self._prefix} Report written -> {path}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    """
    Hash of the raw tensor bytes. Complements the file hash: two .pt files
    can differ byte-for-byte (pickle protocol, save order, torch version)
    while holding identical weights, so a file-hash mismatch alone does not
    prove H3 -- a tensor-hash mismatch does.
    """
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _unwrap_dataset(dataset):
    """
    Follows Subset/ConcatDataset-style .dataset wrappers down to the object
    that actually carries .classes/.class_to_idx/.transform (POKEMON goes
    through random_split, IMAGENET100 does not).
    """
    seen = 0
    while hasattr(dataset, "dataset") and not hasattr(dataset, "classes"):
        dataset = dataset.dataset
        seen += 1
        if seen > 8:  # defensive: never loop on a pathological wrapper chain
            break
    return dataset


def _tensor_stats_line(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().float()
    return (
        f"    {name}: shape={tuple(tensor.shape)} numel={tensor.numel()} "
        f"mean={values.mean().item():+.6e} std={values.std().item():.6e} "
        f"min={values.min().item():+.6e} max={values.max().item():+.6e}"
    )


def _select_probe_tensors(state_dict: dict) -> list[str]:
    """
    Picks PROBE_TENSOR_COUNT weight tensors deterministically, so the local
    and cluster reports fingerprint the same tensors for a given model:
    the preferred names first (conv1.weight), then first/middle/last of the
    >=2-dimensional float tensors in state-dict order (i.e. stem, middle of
    the stack, classifier head).
    """
    weight_names = [
        name for name, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.is_floating_point()
    ]
    if not weight_names:
        return []

    selected = [name for name in PREFERRED_PROBE_TENSORS if name in weight_names]
    for candidate in (weight_names[0], weight_names[len(weight_names) // 2], weight_names[-1]):
        if len(selected) >= PROBE_TENSOR_COUNT:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected[:PROBE_TENSOR_COUNT]


# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------

def _candidate_paths(checkpoint_dir: str | None, load_run_id: str | None, filename: str,
                     run_subdir: str) -> list[str]:
    """
    --checkpoint-dir points at one directory (e.g. the backup *quantized*
    models), but this mode needs both the PTQ/QAT checkpoint and the FP32
    baseline, which live in sibling directories under the same backup root.
    So: try the given directory, then its siblings, then the per-run layout
    results/<run>/<run_subdir>/.
    """
    candidates: list[str] = []
    if checkpoint_dir:
        candidates.append(os.path.join(checkpoint_dir, filename))
        parent = os.path.dirname(os.path.normpath(checkpoint_dir))
        if os.path.isdir(parent):
            for entry in sorted(os.listdir(parent)):
                sibling = os.path.join(parent, entry)
                if os.path.isdir(sibling) and sibling != os.path.normpath(checkpoint_dir):
                    candidates.append(os.path.join(sibling, filename))
    candidates.append(os.path.join(RESULTS_DIR, load_run_id or RUN_ID, run_subdir, filename))
    return candidates


def _resolve_checkpoint(report: _Report, label: str, checkpoint_dir: str | None,
                        load_run_id: str | None, filename: str, run_subdir: str) -> str:
    candidates = _candidate_paths(checkpoint_dir, load_run_id, filename, run_subdir)
    for candidate in candidates:
        if os.path.exists(candidate):
            report.line(f"  {label}: resolved -> {os.path.abspath(candidate)}")
            return candidate
    tried = "\n    ".join(os.path.abspath(c) for c in candidates)
    raise FileNotFoundError(f"[DiagAcc] {label} not found. Tried:\n    {tried}")


# ---------------------------------------------------------------------------
# Check 1 -- checkpoint identity (tests H3)
# ---------------------------------------------------------------------------

def _check_checkpoint_identity(report: _Report, checkpoints: list[tuple[str, str]]) -> None:
    """
    Returns nothing, unlike the other checks: H3 is the one hypothesis a
    single machine cannot settle on its own. These fingerprints only mean
    something next to the cluster's, so this check feeds the report (for
    diffing) rather than the verdict.
    """
    report.section("CHECK 1 -- CHECKPOINT IDENTITY (tests H3: different training run)")
    report.line("Compare every hash below against the cluster copy. Identical file")
    report.line("hashes (or identical per-tensor hashes) rule H3 out; any difference")
    report.line("means the two machines are not evaluating the same weights.")

    for label, path in checkpoints:
        abs_path = os.path.abspath(path)
        size_bytes = os.path.getsize(abs_path)
        mtime = datetime.fromtimestamp(os.path.getmtime(abs_path))

        report.line("")
        report.line(f"{label}")
        report.line(f"  absolute path: {abs_path}")
        report.line(f"  size (bytes):  {size_bytes}")
        report.line(f"  sha256 (file): {_sha256_file(abs_path)}")
        report.line(f"  modified:      {mtime:%Y-%m-%d %H:%M:%S}")

        state_dict = torch.load(abs_path, map_location="cpu", weights_only=True)
        tensor_entries = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
        report.line(f"  state dict entries: {len(state_dict)} "
                    f"({len(tensor_entries)} tensors, "
                    f"{len(state_dict) - len(tensor_entries)} non-tensor)")

        probe_names = _select_probe_tensors(state_dict)
        report.line(f"  probed weight tensors ({len(probe_names)}): {probe_names}")
        for name in probe_names:
            report.line(_tensor_stats_line(name, state_dict[name]))
            report.line(f"      sha256(tensor bytes) = {_sha256_tensor(state_dict[name])}")

        del state_dict


# ---------------------------------------------------------------------------
# Check 2 -- class mapping (tests H1)
# ---------------------------------------------------------------------------

def _describe_class_source(report: _Report, dataset, classes: list[str]) -> dict:
    """
    Reports where the class order actually comes from, and whether that
    source is filesystem-order-dependent. torchvision's ImageFolder derives
    it from DatasetFolder.find_classes, which is `sorted(os.scandir(...))` --
    deterministic across filesystems. A loader that used a bare os.listdir()
    would not be, and that is exactly the H1 failure mode, so the raw
    directory order is printed next to the dataset's order for comparison.
    """
    finding = {"sorted": None, "matches_sorted_listdir": None, "source": "unknown"}

    if isinstance(dataset, datasets.ImageFolder):
        finding["source"] = "torchvision ImageFolder (DatasetFolder.find_classes -> sorted(os.scandir))"
    elif hasattr(dataset, "classes"):
        finding["source"] = f"{type(dataset).__name__}.classes attribute"
    report.line(f"  class-order source: {finding['source']}")

    finding["sorted"] = (classes == sorted(classes))
    report.line(f"  dataset.classes is itself sorted: {finding['sorted']}")

    root = getattr(dataset, "root", None)
    if root and os.path.isdir(root):
        raw_listdir = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]
        report.line(f"  dataset root: {os.path.abspath(root)}")
        report.line(f"  raw os.listdir order (first 5): {raw_listdir[:5]}")
        report.line(f"  dataset.classes      (first 5): {classes[:5]}")
        report.line(f"  raw os.listdir is sorted on this filesystem: "
                    f"{raw_listdir == sorted(raw_listdir)}")
        finding["matches_sorted_listdir"] = (classes == sorted(raw_listdir))
        report.line(f"  dataset.classes == sorted(os.listdir(root)): "
                    f"{finding['matches_sorted_listdir']}")

        if not finding["sorted"]:
            report.line("  >> FLAG: the class list is NOT sorted. Directory order is")
            report.line("  >> filesystem-dependent, so the same image gets a different")
            report.line("  >> integer label on a different machine. This is a likely cause (H1).")

        # A manifest sitting next to the data that the loader ignores is not
        # itself a bug, but it is worth knowing the labels do not come from it.
        parent = os.path.dirname(os.path.normpath(root))
        for manifest_name in MANIFEST_CANDIDATES:
            manifest_path = os.path.join(parent, manifest_name)
            if os.path.exists(manifest_path):
                report.line(f"  NOTE: manifest {manifest_name} exists at {manifest_path}")
                report.line("        but the loader derives classes from directory order, not from it.")
                if manifest_name.endswith(".json"):
                    try:
                        with open(manifest_path, encoding="utf-8") as f:
                            manifest = json.load(f)
                        if isinstance(manifest, dict):
                            report.line(f"        manifest entries: {len(manifest)}; "
                                        f"key set == class set: {set(manifest) == set(classes)}")
                    except (OSError, ValueError) as exc:
                        report.line(f"        (manifest unreadable: {exc})")

    return finding


def _report_split_classes(report: _Report, split_label: str, dataset) -> dict:
    classes = list(getattr(dataset, "classes", []))
    class_to_idx = dict(getattr(dataset, "class_to_idx", {}))

    report.line("")
    report.line(f"--- {split_label} split ---")
    report.line(f"  dataset type: {type(dataset).__name__}")
    report.line(f"  len(dataset.classes): {len(classes)}")

    ordered = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    report.line(f"  class_to_idx, first {CLASS_HEAD_COUNT} by index:")
    for name, idx in ordered[:CLASS_HEAD_COUNT]:
        report.line(f"    {idx:>4} -> {name}")
    report.line(f"  class_to_idx, last {CLASS_TAIL_COUNT} by index:")
    for name, idx in ordered[-CLASS_TAIL_COUNT:]:
        report.line(f"    {idx:>4} -> {name}")

    class_list_hash = _sha256_text("\n".join(classes))
    report.line(f"  sha256(ordered class-name list): {class_list_hash}")

    source = _describe_class_source(report, dataset, classes)
    return {"classes": classes, "hash": class_list_hash, **source}


def _check_class_mapping(report: _Report, val_dataset, train_dataset) -> dict:
    report.section("CHECK 2 -- CLASS MAPPING (tests H1: label mapping differs)")
    report.line("If the sha256 of the ordered class-name list differs from the cluster's,")
    report.line("the two machines assign different integer labels to the same class and")
    report.line("accuracy collapses to roughly chance while every dimension still matches.")

    val_finding = _report_split_classes(report, "VALIDATION", val_dataset)
    train_finding = _report_split_classes(report, "TRAIN", train_dataset)

    report.line("")
    splits_agree = val_finding["hash"] == train_finding["hash"]
    report.line(f"train/val class lists identical (same hash): {splits_agree}")
    if not splits_agree:
        report.line(">> FLAG: train and validation disagree on the class order ON THIS MACHINE.")
        report.line(">> The model was trained against one mapping and is being scored against")
        report.line(">> another. This alone explains a near-chance accuracy (H1).")

    return {
        "val": val_finding,
        "train": train_finding,
        "splits_agree": splits_agree,
        "val_unsorted": val_finding["sorted"] is False,
    }


# ---------------------------------------------------------------------------
# Check 3 -- transform pipeline (tests H2)
# ---------------------------------------------------------------------------

def _transform_stages(transform) -> list:
    if transform is None:
        return []
    return list(getattr(transform, "transforms", [transform]))


def _report_transform(report: _Report, split_label: str, transform) -> dict:
    report.line("")
    report.line(f"--- {split_label} transform ---")
    report.line(f"  repr():")
    for line in repr(transform).splitlines():
        report.line(f"    {line}")

    finding = {"has_normalize": False, "mean": None, "std": None, "geometry": []}

    for stage in _transform_stages(transform):
        stage_name = type(stage).__name__
        if isinstance(stage, transforms.Normalize):
            finding["has_normalize"] = True
            finding["mean"] = tuple(stage.mean)
            finding["std"] = tuple(stage.std)
        elif isinstance(stage, (transforms.Resize, transforms.CenterCrop,
                                transforms.RandomCrop, transforms.RandomResizedCrop)):
            finding["geometry"].append((stage_name, getattr(stage, "size", None)))

    report.line("  resize/crop dimensions:")
    if finding["geometry"]:
        for stage_name, size in finding["geometry"]:
            report.line(f"    {stage_name}: size={size}")
    else:
        report.line("    (none -- images are used at their native resolution)")

    report.line(f"  normalization applied: {finding['has_normalize']}")
    if finding["has_normalize"]:
        report.line(f"    mean tensor: {torch.tensor(finding['mean'])}")
        report.line(f"    std  tensor: {torch.tensor(finding['std'])}")
    else:
        report.line("    >> FLAG: no Normalize in this pipeline. Inputs stay in [0,1] (H2).")

    return finding


def _check_transforms(report: _Report, val_dataset, train_dataset) -> dict:
    report.section("CHECK 3 -- TRANSFORM PIPELINE (tests H2: preprocessing differs)")
    report.line("The validation pipeline must apply the same normalization statistics")
    report.line("training used. Diff these two blocks against the cluster report.")

    val_finding = _report_transform(report, "VALIDATION", getattr(val_dataset, "transform", None))
    train_finding = _report_transform(report, "TRAIN", getattr(train_dataset, "transform", None))

    report.line("")
    stats_agree = (val_finding["mean"] == train_finding["mean"]
                   and val_finding["std"] == train_finding["std"])
    report.line(f"train/val normalization statistics identical: {stats_agree}")
    if not stats_agree:
        report.line(">> FLAG: validation normalizes differently than training did (H2).")

    return {"val": val_finding, "train": train_finding, "stats_agree": stats_agree}


# ---------------------------------------------------------------------------
# Check 4 -- input statistics
# ---------------------------------------------------------------------------

def _classify_input_regime(inputs: torch.Tensor) -> tuple[str, str]:
    """
    Returns (regime_key, human-readable verdict) for what the network is
    actually being fed. With ImageNet mean/std applied, values land roughly
    in [-2.2, +2.7] with per-channel mean near 0 and std near 1; values
    confined to [0,1] mean ToTensor ran but Normalize did not; values above
    1.5 with a [0,255] span mean not even ToTensor's rescale happened.
    """
    minimum = inputs.min().item()
    maximum = inputs.max().item()

    if minimum >= -0.01 and maximum <= 1.01:
        return "unit_interval", ("values confined to [0,1] -- ToTensor ran but NORMALIZATION IS "
                                 "MISSING from the validation path (H2)")
    if minimum >= -0.01 and maximum > 1.5:
        return "byte_range", ("values span roughly [0,255] -- neither ToTensor's rescale nor "
                              "normalization was applied (H2)")
    if minimum < -0.01 and maximum > 1.0:
        return "normalized", "values straddle zero with a >1 span -- consistent with applied normalization"
    return "unexpected", "values match none of the expected regimes -- inspect the pipeline directly"


def _check_input_statistics(report: _Report, loader: DataLoader) -> dict:
    report.section("CHECK 4 -- INPUT STATISTICS (what the network actually receives)")

    inputs, targets = next(iter(loader))
    per_channel_mean = inputs.mean(dim=(0, 2, 3))
    per_channel_std = inputs.std(dim=(0, 2, 3))

    report.line(f"  input shape: {tuple(inputs.shape)}")
    report.line(f"  input dtype: {inputs.dtype}")
    report.line(f"  target shape/dtype: {tuple(targets.shape)} / {targets.dtype}")
    report.line(f"  global min: {inputs.min().item():+.6f}")
    report.line(f"  global max: {inputs.max().item():+.6f}")
    report.line(f"  global mean: {inputs.mean().item():+.6f}")
    report.line(f"  global std:  {inputs.std().item():.6f}")
    report.line("  per-channel mean: " + "  ".join(f"c{i}={v:+.6f}" for i, v in enumerate(per_channel_mean)))
    report.line("  per-channel std:  " + "  ".join(f"c{i}={v:.6f}" for i, v in enumerate(per_channel_std)))

    regime, verdict = _classify_input_regime(inputs)
    report.line("")
    report.line("  Expected under correct ImageNet-style normalization: per-channel mean")
    report.line("  near 0, per-channel std near 1, values roughly in [-2.2, +2.7].")
    report.line(f"  DETECTED: {verdict}")
    if regime != "normalized":
        report.line("  >> FLAG: input regime is wrong; this alone would collapse accuracy (H2).")

    return {
        "regime": regime,
        "per_channel_mean": [v.item() for v in per_channel_mean],
        "per_channel_std": [v.item() for v in per_channel_std],
    }


# ---------------------------------------------------------------------------
# Check 5 -- FP32 baseline evaluation (the decisive split)
# ---------------------------------------------------------------------------

def _evaluate_with_diagnostics(model: nn.Module, loader: DataLoader, num_classes: int,
                               eval_subset_batches: int | None) -> dict:
    """
    One single evaluation pass that produces everything checks 5, 6 and 7
    need (top-1, per-class counts, prediction/label histograms, the first
    batch's sample-level detail). Three separate passes over IMAGENET100's
    5000 validation images at 224x224 on CPU would cost three times as long
    and tell us nothing more.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    batches = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if eval_subset_batches is not None and i >= eval_subset_batches:
                break
            inputs = inputs.to(_model_device(model))
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets.cpu())
            batches += 1

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    correct_mask = preds.eq(targets)

    per_class_total = torch.bincount(targets, minlength=num_classes)
    per_class_correct = torch.bincount(targets[correct_mask], minlength=num_classes)
    pred_histogram = torch.bincount(preds, minlength=num_classes)

    return {
        "preds": preds,
        "targets": targets,
        "num_samples": preds.numel(),
        "num_batches": batches,
        "accuracy": 100.0 * correct_mask.sum().item() / preds.numel() if preds.numel() else float("nan"),
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "pred_histogram": pred_histogram,
    }


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _warn_partial_coverage(report: _Report, result: dict, num_classes: int) -> bool:
    """
    --eval-subset truncates a *class-ordered* validation set, so a handful of
    batches covers a handful of classes -- 2 batches of IMAGENET100 sees 3 of
    100. Checks 6 and 7 both reason about distributions across the label
    space, and are meaningless on that. Says so rather than printing a
    confident-looking verdict computed from three classes.
    """
    present = int((result["per_class_total"] > 0).sum().item())
    if present == num_classes:
        return False
    report.line(f"  >> CAVEAT: only {present}/{num_classes} classes appear in the evaluated")
    report.line("  >> samples, because --eval-subset truncates a class-ordered validation")
    report.line("  >> set. The distributions below are NOT representative -- re-run without")
    report.line("  >> --eval-subset before drawing any conclusion from this check.")
    report.line("")
    return True


def _check_fp32_baseline(report: _Report, baseline_path: str, model_name: str, dataset_name: str,
                         num_classes: int, channels: int, image_size: int,
                         loader: DataLoader, device: torch.device,
                         eval_subset_batches: int | None) -> tuple[dict, dict]:
    from src.model_cnn.train import build_model

    report.section("CHECK 5 -- FP32 BASELINE EVALUATION (the decisive split)")
    report.line("This is the plain FP32 baseline -- not the PTQ/QAT model, not the")
    report.line("PoT-baked model. Interpretation:")
    report.line(f"  * far below ~{EXPECTED_CLUSTER_FP32_ACC:.2f}% -> the fault is in data/preprocessing (H1 or H2)")
    report.line("  * matching it            -> the fault is downstream, in PoT baking or")
    report.line("                              quantized-checkpoint loading")

    model = build_model(
        num_classes=num_classes, model_name=model_name,
        channels=channels, image_size=image_size,
    ).to(device)

    state_dict = torch.load(baseline_path, map_location=device, weights_only=True)
    # main.py's --skip-training path loads strict (the default); loading
    # non-strict here and reporting the counts turns a partially-loaded
    # (i.e. partly random-initialized) model from an exception into a
    # visible finding.
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()

    report.line("")
    report.line(f"  checkpoint: {os.path.abspath(baseline_path)}")
    report.line(f"  build_model({model_name}, num_classes={num_classes}, "
                f"channels={channels}, image_size={image_size}) on {device}")
    report.line(f"  missing keys:    {len(incompatible.missing_keys)} {list(incompatible.missing_keys)[:5]}")
    report.line(f"  unexpected keys: {len(incompatible.unexpected_keys)} {list(incompatible.unexpected_keys)[:5]}")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        report.line("  >> FLAG: the checkpoint does not match the built architecture. Part of")
        report.line("  >> the model is still randomly initialized (H3).")

    report.line("")
    report.line(f"  evaluating on {'all' if eval_subset_batches is None else eval_subset_batches} batches "
                f"({dataset_name} validation split, torch.no_grad, eval mode)...")

    result = _evaluate_with_diagnostics(model, loader, num_classes, eval_subset_batches)

    report.line(f"  samples evaluated: {result['num_samples']} in {result['num_batches']} batches")
    report.line(f"  FP32 baseline top-1 accuracy: {result['accuracy']:.2f}%")
    report.line("")
    report.line(f"  reference -- cluster INT8:  {CLUSTER_INT8_ACC:.2f}%")
    report.line(f"  reference -- cluster FP32:  ~{EXPECTED_CLUSTER_FP32_ACC:.2f}% "
                f"(derived: {CLUSTER_INT8_ACC:.2f} - {CLUSTER_INT8_VS_FP32_DELTA:+.2f})")
    report.line(f"  reference -- local INT8:    {LOCAL_INT8_ACC:.2f}%")
    report.line(f"  reference -- chance level:  {100.0 / num_classes:.2f}%")

    matches_cluster = abs(result["accuracy"] - EXPECTED_CLUSTER_FP32_ACC) <= FP32_MATCH_TOLERANCE_PCT
    report.line("")
    if matches_cluster:
        report.line(f"  CONCLUSION: the FP32 baseline reproduces the cluster figure (within "
                    f"{FP32_MATCH_TOLERANCE_PCT:.0f} pts).")
        report.line("  Data and preprocessing are therefore NOT at fault -- the discrepancy is")
        report.line("  downstream, in PoT baking or quantized-checkpoint loading.")
    else:
        gap = result["accuracy"] - EXPECTED_CLUSTER_FP32_ACC
        report.line(f"  CONCLUSION: the FP32 baseline is {abs(gap):.2f} pts "
                    f"{'BELOW' if gap < 0 else 'ABOVE'} the cluster figure.")
        report.line("  The fault is already present before any quantization: it lies in the")
        report.line("  data/preprocessing path (H1 or H2) or in the checkpoint itself (H3).")

    findings = {"accuracy": result["accuracy"], "matches_cluster": matches_cluster,
                "partial_load": bool(incompatible.missing_keys or incompatible.unexpected_keys)}
    return result, findings


# ---------------------------------------------------------------------------
# Check 6 -- label sanity
# ---------------------------------------------------------------------------

def _check_label_sanity(report: _Report, result: dict, val_dataset, num_classes: int) -> dict:
    report.section("CHECK 6 -- LABEL SANITY (predictions vs. truth)")
    partial_coverage = _warn_partial_coverage(report, result, num_classes)

    classes = list(getattr(val_dataset, "classes", []))

    def class_name(idx: int) -> str:
        return classes[idx] if 0 <= idx < len(classes) else "<out-of-range>"

    preds, targets = result["preds"], result["targets"]

    report.line(f"  first {SAMPLE_ROWS} validation samples:")
    report.line(f"    {'#':>3}  {'pred':>5}  {'true':>5}  {'ok':>3}  {'pred name':<24} {'true name':<24}")
    for i in range(min(SAMPLE_ROWS, preds.numel())):
        p, t = preds[i].item(), targets[i].item()
        report.line(f"    {i:>3}  {p:>5}  {t:>5}  {'Y' if p == t else 'n':>3}  "
                    f"{class_name(p):<24} {class_name(t):<24}")

    true_histogram = torch.bincount(targets, minlength=num_classes)
    top_pred = torch.topk(result["pred_histogram"], k=min(TOP_K_LABELS, num_classes))
    top_true = torch.topk(true_histogram, k=min(TOP_K_LABELS, num_classes))

    report.line("")
    report.line(f"  top-{TOP_K_LABELS} most frequently PREDICTED class indices:")
    for idx, count in zip(top_pred.indices.tolist(), top_pred.values.tolist()):
        report.line(f"    idx {idx:>4} ({class_name(idx):<24}) predicted {count:>5} times")
    report.line(f"  top-{TOP_K_LABELS} most frequent TRUE class indices:")
    for idx, count in zip(top_true.indices.tolist(), top_true.values.tolist()):
        report.line(f"    idx {idx:>4} ({class_name(idx):<24}) appears  {count:>5} times")

    # A balanced validation split makes the "top true" list arbitrary (every
    # class occurs equally often), so the informative signal is how far the
    # *prediction* distribution has collapsed away from that uniformity.
    distinct_predicted = int((result["pred_histogram"] > 0).sum().item())
    top_share = 100.0 * top_pred.values.sum().item() / max(result["num_samples"], 1)
    label_balance = true_histogram.float().std().item()

    report.line("")
    report.line(f"  true-label distribution std across classes: {label_balance:.2f} "
                f"({'balanced' if label_balance < 1.0 else 'imbalanced'} split)")
    report.line(f"  distinct classes ever predicted: {distinct_predicted}/{num_classes}")
    report.line(f"  share of predictions in the top-{TOP_K_LABELS} predicted classes: {top_share:.1f}%")

    # A permuted or offset label mapping shows up as (pred + k) scoring far
    # better than pred itself -- a direct test for the "systematic offset"
    # signature, much sharper than eyeballing the two top-5 lists above.
    report.line("")
    report.line(f"  systematic-offset scan, accuracy of (pred + k) mod {num_classes}:")
    base_accuracy = result["accuracy"]
    best_offset, best_offset_accuracy = 0, base_accuracy
    for k in range(-OFFSET_SCAN_RANGE, OFFSET_SCAN_RANGE + 1):
        shifted = (preds + k) % num_classes
        accuracy = 100.0 * shifted.eq(targets).sum().item() / max(preds.numel(), 1)
        marker = "  <- actual" if k == 0 else ""
        report.line(f"    k={k:+d}: {accuracy:6.2f}%{marker}")
        if accuracy > best_offset_accuracy:
            best_offset, best_offset_accuracy = k, accuracy

    report.line("")
    if best_offset != 0:
        report.line(f"  >> FLAG: shifting predictions by k={best_offset:+d} raises accuracy to "
                    f"{best_offset_accuracy:.2f}% (from {base_accuracy:.2f}%).")
        report.line("  >> That is a systematic index offset between the model's output space")
        report.line("  >> and the loader's labels -- a label-mapping problem (H1).")
    else:
        report.line("  No systematic index offset found: no shift beats the unshifted predictions,")
        report.line("  so a constant-offset label mapping is ruled out. (A non-offset permutation")
        report.line("  would not show up here -- check 7 covers that case.)")

    return {
        "distinct_predicted": distinct_predicted,
        "top_share": top_share,
        "best_offset": best_offset,
        "best_offset_accuracy": best_offset_accuracy,
        "balanced": label_balance < 1.0,
        "partial_coverage": partial_coverage,
    }


# ---------------------------------------------------------------------------
# Check 7 -- per-class accuracy spread
# ---------------------------------------------------------------------------

def _check_per_class_accuracy(report: _Report, result: dict, val_dataset, num_classes: int) -> dict:
    report.section("CHECK 7 -- PER-CLASS ACCURACY SPREAD")
    report.line("Under a label permutation accuracy is near-uniformly low across all")
    report.line("classes. Under genuine model weakness it is uneven -- some classes work.")
    report.line("")
    partial_coverage = _warn_partial_coverage(report, result, num_classes)

    classes = list(getattr(val_dataset, "classes", []))
    totals = result["per_class_total"].float()
    corrects = result["per_class_correct"].float()

    present = totals > 0
    accuracies = torch.zeros_like(totals)
    accuracies[present] = 100.0 * corrects[present] / totals[present]

    present_indices = present.nonzero(as_tuple=True)[0]
    ordered = sorted(present_indices.tolist(), key=lambda i: accuracies[i].item(), reverse=True)

    def class_name(idx: int) -> str:
        return classes[idx] if 0 <= idx < len(classes) else "<out-of-range>"

    report.line("")
    report.line(f"  classes with samples: {len(ordered)}/{num_classes}")
    report.line(f"  best {BEST_WORST_CLASSES} classes:")
    for idx in ordered[:BEST_WORST_CLASSES]:
        report.line(f"    idx {idx:>4} ({class_name(idx):<24}) "
                    f"{accuracies[idx].item():6.2f}%  ({int(corrects[idx])}/{int(totals[idx])})")
    report.line(f"  worst {BEST_WORST_CLASSES} classes:")
    for idx in ordered[-BEST_WORST_CLASSES:]:
        report.line(f"    idx {idx:>4} ({class_name(idx):<24}) "
                    f"{accuracies[idx].item():6.2f}%  ({int(corrects[idx])}/{int(totals[idx])})")

    present_accuracies = accuracies[present]
    best = present_accuracies.max().item() if present_accuracies.numel() else 0.0
    spread_std = present_accuracies.std().item() if present_accuracies.numel() > 1 else 0.0
    zero_classes = int((present_accuracies == 0).sum().item())

    report.line("")
    report.line(f"  per-class accuracy: best {best:.2f}%  "
                f"mean {present_accuracies.mean().item():.2f}%  std {spread_std:.2f}")
    report.line(f"  classes with 0% accuracy: {zero_classes}/{len(ordered)}")

    uniformly_low = best < UNIFORM_LOW_BEST_CLASS_PCT
    report.line("")
    if uniformly_low:
        report.line(f"  PATTERN: near-uniformly low -- not one class clears "
                    f"{UNIFORM_LOW_BEST_CLASS_PCT:.0f}%.")
        report.line("  A model that trained at all gets *some* classes right. This is the")
        report.line("  signature of a broken label mapping or broken inputs (H1/H2), not of a")
        report.line("  weak-but-working model.")
    else:
        report.line(f"  PATTERN: uneven -- the best class reaches {best:.2f}% while others fail.")
        report.line("  The model has genuinely learned class structure, so the labels line up")
        report.line("  with the weights. This argues against H1.")

    return {"best_class_acc": best, "spread_std": spread_std,
            "zero_classes": zero_classes, "uniformly_low": uniformly_low,
            "partial_coverage": partial_coverage}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize(report: _Report, class_findings: dict, transform_findings: dict,
               input_findings: dict, baseline_findings: dict, label_findings: dict,
               spread_findings: dict) -> None:
    report.section("SUMMARY -- WHICH HYPOTHESIS DOES THE EVIDENCE SUPPORT?")

    # Checks 6 and 7 feed the verdict below, so a truncated evaluation makes
    # the verdict itself provisional -- say so at the top, not in a footnote.
    partial_coverage = label_findings["partial_coverage"] or spread_findings["partial_coverage"]
    if partial_coverage:
        report.line("")
        report.line(">> PROVISIONAL: this run evaluated only part of the label space")
        report.line(">> (--eval-subset). Checks 6 and 7 feed the verdict below, so re-run")
        report.line(">> without --eval-subset before acting on it.")

    evidence: list[str] = []
    h1_hits: list[str] = []
    h2_hits: list[str] = []
    h3_hits: list[str] = []

    # ---- H1: label mapping ------------------------------------------------
    if class_findings["val_unsorted"]:
        h1_hits.append("the validation class list is not sorted (filesystem-dependent order)")
    if not class_findings["splits_agree"]:
        h1_hits.append("train and validation disagree on the class order on this machine")
    if label_findings["best_offset"] != 0:
        h1_hits.append(f"shifting predictions by k={label_findings['best_offset']:+d} beats the "
                       f"unshifted accuracy ({label_findings['best_offset_accuracy']:.2f}% vs "
                       f"{baseline_findings['accuracy']:.2f}%)")

    # ---- H2: preprocessing ------------------------------------------------
    if not transform_findings["val"]["has_normalize"]:
        h2_hits.append("the validation pipeline applies no Normalize at all")
    if not transform_findings["stats_agree"]:
        h2_hits.append("validation and training normalize with different statistics")
    if input_findings["regime"] != "normalized":
        h2_hits.append(f"the input regime is '{input_findings['regime']}', not normalized")

    # ---- H3: different checkpoint ----------------------------------------
    if baseline_findings["partial_load"]:
        h3_hits.append("the baseline checkpoint does not fully match the built architecture")

    # ---- The decisive split ----------------------------------------------
    report.line("")
    if baseline_findings["matches_cluster"]:
        evidence.append(
            f"The FP32 baseline reproduces the cluster figure "
            f"({baseline_findings['accuracy']:.2f}% vs ~{EXPECTED_CLUSTER_FP32_ACC:.2f}%), so data and "
            f"preprocessing are exonerated and the fault is downstream (PoT baking or "
            f"quantized-checkpoint loading)."
        )
        verdict = ("NEITHER H1 NOR H2 NOR H3 -- the fault is downstream of the FP32 baseline. "
                   "Investigate bake_pot_into_standard_layers and the PTQ/QAT checkpoint load "
                   "path next, not the dataset.")
    else:
        evidence.append(
            f"The FP32 baseline is already wrong ({baseline_findings['accuracy']:.2f}% vs an expected "
            f"~{EXPECTED_CLUSTER_FP32_ACC:.2f}%), so the fault predates quantization entirely."
        )
        if h2_hits:
            verdict = ("H2 (transform/preprocessing). Fix the validation preprocessing first; "
                       "it is broken independently of anything else here.")
        elif h1_hits:
            verdict = ("H1 (class-to-index label mapping). The model's output space and the "
                       "loader's labels do not line up.")
        elif spread_findings["uniformly_low"]:
            verdict = ("H1 or H3. Nothing observable is wrong with the class ordering or the "
                       "transforms, yet no class works at all -- which happens either under a "
                       "non-offset label permutation (H1) or because these weights never learned "
                       "this label space (H3). Compare check 1's tensor hashes against the "
                       "cluster copies to settle it; that comparison is the only remaining "
                       "discriminator.")
        else:
            verdict = ("H3 (different training run). Preprocessing and label mapping both check "
                       "out, and per-class accuracy is uneven -- so these weights work, they are "
                       "just not the weights that produced the cluster number. Compare check 1's "
                       "hashes against the cluster copies.")
            # H3 is the residual hypothesis: nothing observable on one machine
            # can confirm it, so its bucket would otherwise read "no evidence"
            # directly above a verdict of H3. Say why instead.
            h3_hits.append("reached by elimination -- H1 and H2 are both excluded above, and this "
                           "machine alone cannot see whether the cluster held different weights; "
                           "check 1's hashes are the only way to confirm it")

    for label, hits in (("H1 (label mapping)", h1_hits),
                        ("H2 (transform pipeline)", h2_hits),
                        ("H3 (different checkpoint)", h3_hits)):
        report.line("")
        report.line(f"{label}: {len(hits)} supporting observation(s)")
        for hit in hits:
            report.line(f"  - {hit}")
        if not hits:
            report.line("  - none; this mode found no evidence for it")

    report.line("")
    report.line(f"Per-class pattern: "
                f"{'near-uniformly low' if spread_findings['uniformly_low'] else 'uneven'} "
                f"(best class {spread_findings['best_class_acc']:.2f}%, "
                f"{spread_findings['zero_classes']} classes at 0%).")
    report.line(f"Prediction spread: {label_findings['distinct_predicted']} distinct classes ever "
                f"predicted, {label_findings['top_share']:.1f}% of predictions in the top-"
                f"{TOP_K_LABELS} classes.")

    report.line("")
    for item in evidence:
        report.line(item)
    report.line("")
    report.line(f"VERDICT: {verdict}")
    report.line("")
    report.line("Next step regardless of verdict: run this same mode on the cluster and diff the")
    report.line("two report files. Checks 1-4 are pure fingerprints -- any line that differs")
    report.line("between the two machines is, by construction, a cause of the accuracy gap.")


# ---------------------------------------------------------------------------
# Diagnostic loader
# ---------------------------------------------------------------------------

def _build_diag_loader(report: _Report, val_loader: DataLoader) -> DataLoader:
    """
    Rewraps the *existing* dataset (never a freshly constructed one) with the
    settings this diagnosis requires: num_workers=0 and pin_memory=False,
    since pinning is a GPU-transfer optimization that previously contributed
    to an OOM here, plus a capped batch size. None of these affect top-1
    accuracy; the original loader's settings are printed alongside so any
    difference stays visible in the report.
    """
    batch_size = min(val_loader.batch_size or DIAG_MAX_BATCH_SIZE, DIAG_MAX_BATCH_SIZE)
    report.line(f"  original val loader: batch_size={val_loader.batch_size} "
                f"num_workers={val_loader.num_workers} pin_memory={val_loader.pin_memory}")
    report.line(f"  diagnostic loader:   batch_size={batch_size} num_workers=0 "
                f"pin_memory=False shuffle=False (same dataset object)")
    return DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_acc_mismatch_diagnosis(
    model_name: str = DEFAULT_MODEL,
    dataset_name: str = DEFAULT_DATASET,
    stage: str = DEFAULT_STAGE,
    checkpoint_dir: str | None = None,
    load_run_id: str | None = None,
    eval_subset: int | None = None,
) -> str:
    """
    Runs all seven checks in order and writes the combined report to
    results/<RUN_ID>/logs/acc_mismatch_diagnosis.txt. Returns that path.
    """
    report = _Report()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report.line("ACCURACY-MISMATCH DIAGNOSIS")
    report.line("=" * 78)
    report.line(f"generated:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    report.line(f"run id:     {RUN_ID}")
    report.line(f"scope:      {model_name} / {dataset_name} / {stage}")
    report.line(f"device:     {device} (cuda available: {torch.cuda.is_available()})")
    report.line(f"host:       {os.uname().nodename}")
    report.line(f"torch:      {torch.__version__}")
    report.line("")
    report.line(f"Explaining: local {LOCAL_INT8_ACC:.2f}% vs cluster {CLUSTER_INT8_ACC:.2f}% top-1 on the")
    report.line(f"same checkpoint. INT8-vs-FP32 delta is {CLUSTER_INT8_VS_FP32_DELTA:+.2f}%, so the FP32")
    report.line("baseline itself differs -- quantization is not the cause.")

    specs = DATASET_SPECS[dataset_name]
    num_classes = specs["num_classes"]
    channels, image_size = specs["channels"], specs["image_size"]

    # ---- checkpoints ------------------------------------------------------
    report.line("")
    report.line("Resolving checkpoints:")
    baseline_path = _resolve_checkpoint(
        report, "FP32 baseline", checkpoint_dir, load_run_id,
        f"baseline_{model_name}_{dataset_name}_float32.pt", "models",
    )
    stage_path = _resolve_checkpoint(
        report, f"{stage} PoT checkpoint", checkpoint_dir, load_run_id,
        f"{stage.lower()}_po2_{model_name}_{dataset_name}.pt", "quantized_models",
    )

    # ---- data (the existing loader, unmodified) ---------------------------
    report.line("")
    report.line(f"Loading {dataset_name} through the existing get_data_loaders() path:")
    train_loader, val_loader, loader_num_classes = get_data_loaders(dataset_name)
    val_dataset = _unwrap_dataset(val_loader.dataset)
    train_dataset = _unwrap_dataset(train_loader.dataset)

    report.line(f"  train samples: {len(train_loader.dataset)}  "
                f"val samples: {len(val_loader.dataset)}")
    report.line(f"  num_classes from loader: {loader_num_classes}  "
                f"from DATASET_SPECS: {num_classes}")
    if loader_num_classes != num_classes:
        report.line("  >> FLAG: the loader and DATASET_SPECS disagree on the class count.")
    diag_loader = _build_diag_loader(report, val_loader)

    # ---- the seven checks -------------------------------------------------
    _check_checkpoint_identity(report, [
        ("FP32 baseline", baseline_path),
        (f"{stage} PoT checkpoint", stage_path),
    ])
    class_findings = _check_class_mapping(report, val_dataset, train_dataset)
    transform_findings = _check_transforms(report, val_dataset, train_dataset)
    input_findings = _check_input_statistics(report, diag_loader)
    eval_result, baseline_findings = _check_fp32_baseline(
        report, baseline_path, model_name, dataset_name,
        num_classes, channels, image_size, diag_loader, device, eval_subset,
    )
    label_findings = _check_label_sanity(report, eval_result, val_dataset, num_classes)
    spread_findings = _check_per_class_accuracy(report, eval_result, val_dataset, num_classes)

    _summarize(report, class_findings, transform_findings, input_findings,
               baseline_findings, label_findings, spread_findings)

    report_path = os.path.join(LOG_DIR, REPORT_FILENAME)
    report.write(report_path)
    return report_path
