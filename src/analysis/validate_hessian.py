import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compare_estimators(
    custom_traces: dict[str, float],
    pyhessian_traces: dict[str, float],
    rel_tol: float = 0.10,
) -> None:
    """
    Prints a per-layer comparison of two trace dictionaries and flags layers
    whose relative difference exceeds rel_tol. Both estimators are stochastic,
    so a few percent variation is expected; large gaps indicate a real bug.
    """
    shared = sorted(set(custom_traces) & set(pyhessian_traces))
    if not shared:
        logger.warning("No overlapping layers to compare.")
        return

    header = f"{'layer':<28}{'custom':>14}{'pyhessian':>14}{'rel_diff':>12}"
    print(header)
    print("-" * len(header))
    for name in shared:
        a = custom_traces[name]
        b = pyhessian_traces[name]
        rel = abs(a - b) / (abs(b) + 1e-6)
        flag = "  <-- CHECK" if rel > rel_tol else ""
        print(f"{name:<28}{a:>14.4f}{b:>14.4f}{rel:>12.2%}{flag}")

    # Whole-model agreement: the sum of per-layer traces should also match.
    sum_a = sum(custom_traces[n] for n in shared)
    sum_b = sum(pyhessian_traces[n] for n in shared)
    sum_rel = abs(sum_a - sum_b) / (abs(sum_b) + 1e-6)
    print("-" * len(header))
    print(f"{'SUM':<28}{sum_a:>14.4f}{sum_b:>14.4f}{sum_rel:>12.2%}")


if __name__ == "__main__":
    # Minimal self-contained sanity run on a tiny model so the harness itself
    # can be exercised without the full project. Replace with your real model,
    # loader, and criterion for the actual validation.
    logging.basicConfig(level=logging.INFO)

    from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    ).to(device)

    # Fake one batch of data as a stand-in loader.
    x = torch.randn(64, 3, 8, 8)
    y = torch.randint(0, 10, (64,))
    loader = [(x, y)]

    criterion = nn.CrossEntropyLoss()

    py = compute_layerwise_hessian_trace_pyhessian(
        model, loader, criterion, device, num_batches=1
    )
    print("\nPyHessian per-layer traces:")
    for k, v in py.items():
        print(f"  {k}: {v:.4f}")