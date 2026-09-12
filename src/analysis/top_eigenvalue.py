import logging

import torch
import torch.nn as nn
from pyhessian import hessian
from src.analysis.pyhessian import _single_batch, _target_layers

logger = logging.getLogger(__name__)


def compute_top_eigenvalue(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 5,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> dict[str, float]:
    """
    Per-layer top Hessian eigenvalue via PyHessian's power-iteration engine.

    Mirrors compute_layerwise_hessian_trace_pyhessian: one PyHessian object
    per target layer, with the engine's param/grad lists restricted to that
    layer's weight so power iteration runs block-diagonally (lambda_max of
    H_ii). Must receive an unwrapped (non-DDP) model, since PyHessian calls
    loss.backward(create_graph=True) and DDP's all_reduce grad hooks break it.

    Reference: PyHessian eigenvalues() API
    https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
    """
    model.eval()
    batch = _single_batch(dataloader, num_batches, device)
    if batch is None:
        logger.warning("Empty dataloader; cannot compute top eigenvalue.")
        return {}

    inputs, targets = batch
    target_layers = _target_layers(model)
    if not target_layers:
        logger.warning("No conv/linear weight parameters found.")
        return {}

    eigenvalues: dict[str, float] = {}
    use_cuda = device.type == "cuda"

    for name, param in target_layers.items():
        hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=use_cuda)

        try:
            idx = next(i for i, p in enumerate(hessian_comp.params) if p is param)
        except StopIteration:
            logger.warning(f"Layer '{name}' not found in engine params; skipping.")
            continue

        hessian_comp.params = [hessian_comp.params[idx]]
        hessian_comp.gradsH = [hessian_comp.gradsH[idx]]

        top_values, _ = hessian_comp.eigenvalues(maxIter=max_iter, tol=tol, top_n=1)
        eigenvalues[name] = float(top_values[-1])
        logger.info(f"[PyHessian] {name}: lambda_max={eigenvalues[name]:.4f}")

        model.zero_grad(set_to_none=True)
        del hessian_comp
        if use_cuda:
            torch.cuda.empty_cache()

    return eigenvalues