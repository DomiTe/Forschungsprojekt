"""
_seed_stats.py -- shared seed-derivation and cross-seed aggregation for the
--n-seeds variance pass (relock_traces.py, quant_induced_trace.py,
random_init_control.py, diagnose_activations.py, spike_layer_cause.py).

Every affected phase derives its seed list via derive_seeds(base_seed,
n_seeds), so one --n-seeds/--base-seed pair means the same thing everywhere.
The drift diagnostic (weight_ablation_diagnose.py) established the analysis
code is bit-exact deterministic given a fixed seed -- multi-seed here
therefore estimates only genuine stochastic sources (Hutchinson probe draws,
batch selection), not measurement noise.

IMPORTANT for anyone verifying --n-seeds 1 against a pre-existing saved CSV:
the phases this pass touches previously used their OWN hardcoded probe-seed
constants (e.g. relock_traces.CANONICAL_PROBE_SEED = 20260811,
random_init_control.PROBE_SEED = 20260810), not a shared default. With
--base-seed defaulting to 42, a bare --n-seeds 1 run will NOT numerically
match an old saved CSV (different seed value in, different Hutchinson probe
draw out) -- the code PATH is a no-op at n_seeds=1 (identical single-
iteration call sequence), but the NUMBERS only match if you also pass the
historical seed explicitly, e.g. --base-seed 20260811 --n-seeds 1 for
relock-traces/quant-induced-trace/spike-layer-cause, or --base-seed 20260810
for random-init-control's probe seed (its init-seed axis has no single
historical constant to match -- see random_init_control.py's own note).
Verify this bit-exact-at-matching-seed property before trusting any
multi-seed output; this module does not run anything itself.
"""

import math
import statistics


def derive_seeds(base_seed: int, n_seeds: int) -> list[int]:
    assert n_seeds >= 1, f"n_seeds must be >= 1, got {n_seeds}"
    return [base_seed + i for i in range(n_seeds)]


def aggregate(values: list[float]) -> tuple[float, float]:
    """
    Returns (mean, std) across seeds for one layer/metric. std is the
    sample std (ddof=1) when len(values) >= 2, else 0.0 -- a single sample
    has no estimable spread, and reporting NaN there would read as a
    missing-data gap rather than "not enough seeds to estimate variance,"
    which is what it actually is.

    Metrics fed in here (e.g. outlier_factor) deliberately use inf as a
    sentinel for "act_p99 == 0", so this can't delegate to
    statistics.stdev: its exact-Fraction internals raise AttributeError as
    soon as one value is inf/nan (mss ends up a bare float, which has no
    .numerator). Computed by hand instead -- mean/std of a list containing
    inf/nan naturally comes out inf/nan rather than crashing.
    """
    if not values:
        return float("nan"), float("nan")
    if any(not math.isfinite(v) for v in values):
        mean = statistics.mean(values)
        return mean, float("nan") if len(values) >= 2 else 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std
