import copy
import logging

import torch
import torch.nn as nn
from pyhessian import hessian

logger = logging.getLogger(__name__)


def _target_layers(model: nn.Module) -> dict[str, nn.Parameter]:
    # Restrict to 2D/4D weight tensors (Conv2d, Linear); skip biases and BN.
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and "weight" in name and param.dim() >= 2
    }

def _disable_inplace_relu(model: nn.Module) -> None:
    # inplace ReLU breaks create_graph=True double-backward; make them out-of-place
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            
def _single_batch(dataloader, num_batches: int, device: torch.device):
    # PyHessian's single-batch path reuses one gradient graph, which is the
    # cheapest correct mode. We concatenate a few batches into one so the
    # trace is estimated over more data (HAWQ-V2 uses >=512 points).
    inputs_list, targets_list = [], []
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs_list.append(inputs)
        targets_list.append(targets)
    if not inputs_list:
        return None
    inputs = torch.cat(inputs_list, dim=0).to(device)
    targets = torch.cat(targets_list, dim=0).to(device)
    return inputs, targets


def compute_layerwise_hessian_trace_pyhessian(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 5,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> dict[str, float]:
    """
    Per-layer (block-diagonal) Hessian trace using PyHessian's HVP engine.

    For each target layer, a PyHessian object is built and its parameter set is
    overwritten with that single layer's weight, so the double-backward is
    restricted to one layer. This yields Tr(H_ii) without cross-layer variance.
    Convergence-based stopping (max_iter, tol) matches PyHessian's trace().

    Reference: Yao et al., "PyHessian" (2020); Dong et al., "HAWQ-V2" (NeurIPS 2020).
    """
    model.eval()
    _disable_inplace_relu(model)
    batch = _single_batch(dataloader, num_batches, device)
    if batch is None:
        logger.warning("Empty dataloader; cannot compute Hessian trace.")
        return {}

    inputs, targets = batch
    targets_layers = _target_layers(model)
    if not targets_layers:
        logger.warning("No conv/linear weight parameters found.")
        return {}

    traces: dict[str, float] = {}
    use_cuda = device.type == "cuda"

    for name, param in targets_layers.items():
        # Fresh engine per layer. __init__ runs one create_graph backward and
        # populates self.params / self.gradsH from all requires_grad params;
        # we then restrict both to this layer's weight only.
        hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=use_cuda)
        print([p.shape for p in hessian_comp.params])
        # Locate this layer's gradient within the engine's ordered lists.
        # get_params_grad preserves model.parameters() order, so we match by identity.
        try:
            idx = next(
                i for i, p in enumerate(hessian_comp.params) if p is param
            )
        except StopIteration:
            logger.warning(f"Layer '{name}' not found in engine params; skipping.")
            continue

        hessian_comp.params = [hessian_comp.params[idx]]
        hessian_comp.gradsH = [hessian_comp.gradsH[idx]]

        trace_list = hessian_comp.trace(maxIter=max_iter, tol=tol)
        # trace() returns the running list of v^T H v; the estimate is its mean.
        traces[name] = float(sum(trace_list) / len(trace_list))
        logger.info(f"[PyHessian] {name}: trace={traces[name]:.4f} "
                    f"({len(trace_list)} iters)")
        
        model.zero_grad(set_to_none=True)
        del hessian_comp
        if use_cuda:
            torch.cuda.empty_cache()

    return traces