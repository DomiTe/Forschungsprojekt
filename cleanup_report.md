# Pipeline Cleanup Report

Branch: `cleanup/pipeline`, based on `FSP-B/quant-trace` @ `6a55e7e`.
Scope: Part 1 remove/keep list, mechanical only. Zero science change — see
Verification below.

## Preconditions (checked before starting)

- **Working tree.** Was dirty at the start of this task: (a) this session's
  own uncommitted work (`spike_layer_cause.py`, `weight_ablation_diagnose.py`,
  and the `--damage-mode` changes to `weight_ablation_canonical.py`/`main.py`/
  `helper.py` — exactly the flags this cleanup's Part 1 lists as "keep"), and
  (b) unrelated, not-mine changes under `results/backup_models/` (checkpoint
  binaries). Per user instruction: (a) was committed to `FSP-B/quant-trace`
  as `6a55e7e` before branching; (b) was left exactly as-is, uncommitted —
  it rides along in the working tree regardless of branch (git doesn't
  isolate untracked/uncommitted changes from a branch switch) but nothing
  under `results/` was touched by this cleanup.
- **Canonical run archived under a fixed RUN_ID.** Not verified as a
  single blessed RUN_ID — this repo has several `results/<RUN_ID>/csv/
  weight_ablation_canonical*.csv` outputs from different sessions, no
  marker distinguishing "the" canonical one. This did not block the
  mechanical work (nothing under `results/` is touched by this cleanup),
  but it's worth the user's attention before the paper's numbers are
  finalized from a specific run.
- **`--n-seeds` / `--base-seed` (seeded variance pass).** **These flags do
  not exist anywhere in the codebase.** `helper.py` is the sole
  `parse_args()` definition and has no such arguments; the only related
  symbol is an internal, non-CLI `n_seeds` parameter inside
  `random_init_control.py` (a different thing — it's a column name for
  random-init trace averaging, not a general variance-pass CLI mode). There
  was nothing to "verify still respects the phases we intend to keep"
  because the flags aren't wired up. Flagged for the user; not implemented
  here (out of scope — "no new functionality").

## Part 2 — Preservation checks

### Check 1: import graph

`grep -rn "from src.analysis.<module> import" src/` for each removal
candidate, before touching anything:

- **`weight_ablation.py`**: imported by **five** retained modules, not just
  the two named in the task text — `quant_induced_trace.py`,
  `relock_traces.py`, `weight_ablation_canonical.py`, `spike_layer_cause.py`,
  and `weight_ablation_diagnose.py`. Symbols actually imported (union across
  all five): `_resolve_checkpoint_robust`, `WeightAblationCheckpointError`,
  `_run_part0`, `_verify_weight_mask`, `_build_eval_loader`,
  `WeightAblationError`. All six extracted verbatim into a new
  `src/analysis/_ablation_common.py`; all five importers repointed.
  `weight_ablation.py`'s own P1-specific symbols (`_run_isolation_sweep`,
  `_run_correlation`, `run_weight_ablation`, `ABLATION_FIELDNAMES`,
  `CORRELATION_FIELDNAMES`) were imported by nobody else and were removed
  with the module.
- **`validate_pot.py`**: imported only by `main.py` (the branch being
  removed). `src/quantization/deploy.py` *mentions* it in a docstring
  (historical motivation for why `deploy.build_int8_model` exists — the bug
  validate_pot.py helped catch) but never imports from it. Left that
  historical mention as-is (it's accurate prose about a past event, not a
  broken import claim).
- **`diagnose_acc.py`**: imported only by `main.py` (the branch) and
  `helper.py` (the `DEFAULT_MODEL`/`DEFAULT_DATASET`/`DEFAULT_STAGE`
  defaults for the `--diag-*` flags). Both removed together; no other
  retained module touches it.

No `_common` extraction was needed for `validate_pot.py` or `diagnose_acc.py`
— clean, self-contained removals.

### Check 2: CSV producers/consumers

Enumerated every `"*.csv"` string literal across `src/` and its producing
file. No retained flag reads a CSV that only a removed flag produced:
`weight_ablation.csv` / `weight_ablation_correlation.csv` (P1's own outputs)
become orphaned but nothing downstream consumed them. One near-miss:
`spike_layer_cause.py`'s damage-CSV auto-discovery globs for a legacy
filename `weight_ablation_canonical.csv` (no `_v2` suffix) — this is
intentional backward-compatible discovery of an already-archived,
committed investigation CSV from before the `_v2` naming convention, not a
producer/consumer break (nothing needs to produce that exact name going
forward).

### Check 3: test coverage

No test suite exists anywhere in the repo (`find . -iname "test_*.py"` and
a `def test_` / `import pytest` / `import unittest` grep across every `.py`
file, excluding `.venv`, both return zero matches). Nothing to remove or
split.

### Check 4: repo-wide grep, before and after

Run over the whole repo (not just `src/`), `.venv` and `results/` logs
excluded from the "must fix" set (frozen historical run logs correctly
still mention these flags — they ran at the time; per "do not delete
anything under `results/`" they were left untouched).

| Flag / module | Before (non-results hits) | After |
|---|---|---|
| `--weight-ablation` (word-boundary) | `weight_ablation.py` (module docstring + `helper.py` registration) | 0 |
| `--validate-pot-int8` / `validate_pot` | `main.py`, `validate_pot.py`, `deploy.py` (prose), `helper.py` | `deploy.py` (prose only, left — see Check 1) |
| `--diagnose-acc-mismatch` / `--diag-*` / `diagnose_acc` | `main.py`, `helper.py` (×2: registration + `--eval-subset`'s help text), `diagnose_acc.py` | 0 |

One real fix needed beyond deleting the registrations: `--eval-subset`'s
help text listed `--diagnose-acc-mismatch` as a consumer (`"For
--deploy-cpu-fbgemm and --diagnose-acc-mismatch: ..."`) — fixed to name only
`--deploy-cpu-fbgemm`, its one remaining consumer.

No README, docs, or notebook (`notebooks/*.ipynb`, checked cell-by-cell)
mentions any removed flag.

Six docstring/comment cross-references to `weight_ablation.py` as an import
source (in `quant_induced_trace.py`, `spike_layer_cause.py` ×1,
`weight_ablation_canonical.py` ×2, `weight_ablation_diagnose.py` ×1,
`relock_traces.py` ×1) were factually wrong after the `_ablation_common.py`
extraction and were corrected to point at the new module. Two other
mentions of `weight_ablation.py` (as *historical* motivation — "the original
pre-relock weight_ablation.py sweep found...") were left alone; they
describe a past measurement, not an import path, and remain true.

## Ask-before-touching items — investigated, not removed

- **`--deploy-int8-only`**: read `_run_benchmark_int8_only`'s own code and
  comments. It does **not** load anything `_run_deploy_int8_only` saves
  (`deployed_{weightonly,full}_*.pt`) — it explicitly rebuilds the int8
  model fresh via `deploy.build_int8_model`, with a comment stating this is
  deliberate ("an earlier version of this code did [reload], and silently
  ended up benchmarking an fp32 model under the 'int8' label"). So there is
  no *file* dependency. There is a *prose* one: `_run_benchmark_int8_only`'s
  comment says it uses "the same path Deploy-Int8-Only proved preserves
  accuracy" — i.e. `--deploy-int8-only`'s own internal round-trip check
  (reload the model it just saved, confirm accuracy matches) is what
  originally validated that reconstruction path. **Recommendation:** likely
  removable (no code dependency), but the validation-methodology link is
  real enough that I did not mark it a clean orphan. Left untouched, not
  removed.
- **`--ablate-layer-quantization` / `--ablate-top-k` / `--ablate-layers`**:
  confirmed `layer_ablation.py` imports nothing from `weight_ablation.py`,
  `validate_pot.py`, or `diagnose_acc.py` — fully independent of everything
  removed. Matches the task's own characterization (superseded by the
  canonical PoT ablation, but historically load-bearing for the
  activation-quant investigation's motivation). **Recommendation:**
  candidate for removal in a *future*, explicitly-approved pass once the
  paper's methods section no longer needs to cite the "excluding conv1
  recovers 3.07 pts" fbgemm-path number. Left untouched, not removed.

## Found during the check, not on any list — flagged, not touched

- **`src/quantization/real_quant_attempt/`** (4 files: `wrapper.py`,
  `convert.py`, `export.py`, `kernel.py`): imported by nothing in the
  codebase — confirmed via `grep -rln "real_quant_attempt" src/` returning
  only self-matches. Entirely orphaned, but never named in Part 1's
  remove/keep/ask lists, so left untouched per "do not remove any flag
  [or, by the same logic, module] not explicitly listed."
- **`src/analysis/hessian.py`'s `compute_layerwise_hessian_trace`** is
  imported by `main.py` (line 38) but **never called** — only
  `pyhessian.py`'s `compute_layerwise_hessian_trace_pyhessian` is actually
  invoked in the training pipeline. This looks like a dead import left over
  from before the pipeline switched estimators. Not on the remove list, so
  the import itself was left in `main.py`; flagged here for a future pass.
- **A full ImageNet-1000 loader, commented out** in both
  `src/utility/utils.py` (`_get_imagenet_loaders`, ~38 lines) and mirrored
  by a commented `"IMAGENET": {...}` entry in `src/utility/config.py`'s
  `DATASET_SPECS`. Judged this a deliberate paused-feature scaffold (still
  listed as a recognized `DATASET_NAME` option in `config.py`'s comment),
  not abandoned debris, and left it alone — see the different call made on
  `hessian.py`'s commented block below.

## Mechanical removal

Deleted: `src/analysis/weight_ablation.py`, `src/analysis/validate_pot.py`,
`src/analysis/diagnose_acc.py`.

Added: `src/analysis/_ablation_common.py` (the six shared symbols above,
moved verbatim — see Check 1).

Removed from `src/main.py`: the `--weight-ablation`, `--validate-pot-int8`,
and `--diagnose-acc-mismatch` branches in `main()`; the
`_run_validate_pot_int8` function and its `VALIDATE_POT_DATASETS`/
`VALIDATE_POT_STAGES` scope constants; the `from src.analysis import
validate_pot` import.

Removed from `src/utility/helper.py`: the `--weight-ablation`,
`--validate-pot-int8`, `--diagnose-acc-mismatch`, `--diag-model`,
`--diag-dataset`, `--diag-stage` argument registrations; the deferred
`from src.analysis.diagnose_acc import DEFAULT_MODEL, ...` block at the top
of `parse_args()`; the now-dead `--diagnose-acc-mismatch` mention in
`--eval-subset`'s help text.

Repointed five importers (`quant_induced_trace.py`, `relock_traces.py`,
`weight_ablation_canonical.py`, `spike_layer_cause.py`,
`weight_ablation_diagnose.py`) from `weight_ablation` to `_ablation_common`,
plus their docstring cross-references (Check 4).

## Verification: canonical-run zero-diff

Command: `--weight-ablation-canonical --checkpoint-dir
results/20260810_104157_31209/quantized_models --canonical-traces-csv
results/relock_traces_investigation_1786438687/csv/canonical_traces.csv`
(full scope: resnet18 + resnet50, PTQ + QAT — not narrowed to resnet50 PTQ
only, for a stronger check than the suggested minimum).

Run once immediately before the mechanical removal (still importing the
soon-to-be-deleted `weight_ablation.py`), and once immediately after
(importing only `_ablation_common.py`), same checkpoints, same seed.

```
$ diff before/weight_ablation_canonical_v2.csv              after/weight_ablation_canonical_v2.csv
$ diff before/weight_ablation_canonical_correlation_v2.csv  after/weight_ablation_canonical_correlation_v2.csv
(both empty -- zero diff)

$ md5sum before/*.csv after/*.csv
95767f69d8f0c6252a1880b9f52e2142  weight_ablation_canonical_v2.csv              (before)
050091418d3b501e721efc24f1715615  weight_ablation_canonical_correlation_v2.csv  (before)
95767f69d8f0c6252a1880b9f52e2142  weight_ablation_canonical_v2.csv              (after)
050091418d3b501e721efc24f1715615  weight_ablation_canonical_correlation_v2.csv  (after)
```

Bit-exact identical. **Acceptance criterion met.**

## Part 3 — Comment tightening (retained code)

Method: pattern-matched the whole retained `src/` tree for the specific
violation classes (`# TODO` without owner/date, change-history phrasing
["used to", "previously", "v1/v2" as a *code*-version marker, "no longer"],
`#####`-style dividers, bare `assert` without a message, 3+ line
commented-out code blocks), then read the flagged hits in context and fixed
genuine violations. Also read every file this cleanup itself touched
(`_ablation_common.py`, the five repointed importers, `main.py`,
`helper.py`) end-to-end, plus a sample of previously-unread retained files
(`quantizer.py`, `model_cnn/resnet18.py`, `model_cnn/pretrained_resnet18.py`)
to spot-check general comment quality outside the pattern search.

Findings:
- **`v1`/`v2` mentions in `weight_ablation_diagnose.py`** matched the
  change-history regex but are not code-history — they're the literal
  names of the two *experimental runs* the whole module exists to compare
  (matches the `_v2` suffix already in the CSV filenames throughout this
  codebase). Kept.
- **`src/analysis/hessian.py`**: removed a ~35-line commented-out block (a
  draft alternate implementation of the exact same Hutchinson trace the
  active code above it already computes — pure abandoned exploration, not
  a documented alternative) and one pure-restatement comment (`# Forward
  pass` immediately above `outputs = model(inputs)`).
- No bare `assert` without a message anywhere in retained, wired-in code
  (the only hits were in the orphaned, unimported `real_quant_attempt/`,
  out of scope — see above).
- No `#####` dividers, no un-owned `# TODO` comments anywhere in the
  retained tree.
- Everything else sampled (quantizer.py, the resnet definitions,
  main.py's per-mode header comments) was already consistent with the
  WHY-focused style these rules ask for — no changes needed.

This was **not** an exhaustive line-by-line read of every retained file
(the retained tree spans `model_cnn/`, `quantization/`, `utility/`, and a
dozen `analysis/` modules); it was a systematic search for the specific
violation patterns plus close reading of everything this cleanup itself
touched. Flagging this explicitly rather than claiming a full manual audit.

## Files changed

```
 A  src/analysis/_ablation_common.py
 D  src/analysis/diagnose_acc.py
 M  src/analysis/hessian.py
 M  src/analysis/quant_induced_trace.py
 M  src/analysis/relock_traces.py
 M  src/analysis/spike_layer_cause.py
 D  src/analysis/validate_pot.py
 D  src/analysis/weight_ablation.py
 M  src/analysis/weight_ablation_canonical.py
 M  src/analysis/weight_ablation_diagnose.py
 M  src/main.py
 M  src/utility/helper.py
```

`results/` untouched throughout (per constraint) — including the
pre-existing, unrelated `results/backup_models/` diff, which is still
sitting uncommitted in the working tree exactly as it was found.
