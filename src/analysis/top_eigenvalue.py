import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def compute_top_eigenvalue(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 5,
    num_iterations: int = 20,  # power iteration steps
) -> dict:
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    target_params = {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and "weight" in name and param.dim() >= 2
    }

    eigenvalues = {name: 0.0 for name in target_params}

    for name, param in target_params.items():
        # random unit vector to start power iteration
        v = torch.randn_like(param)
        v = v / v.norm()


        for _ in range(num_iterations):
            v_dot_Hv = 0.0
            hv_accumulated = torch.zeros_like(param)

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                grad = torch.autograd.grad(
                    loss, param, create_graph=True, retain_graph=True
                )[0]

                v_dot_g = torch.sum(grad * v)
                hv = torch.autograd.grad(
                    v_dot_g, param, retain_graph=False
                )[0]
                hv_accumulated += hv
                v_dot_Hv += torch.sum(v * hv).item()

                del grad, hv, outputs, loss
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            # normalize to get new eigenvector estimate
            v = hv_accumulated / (hv_accumulated.norm() + 1e-8)
            # v = v * (1.0 / (abs(v_dot_Hv) + 1e-8))
            # v = v / v.norm()

        eigenvalues[name] = v_dot_Hv / num_batches

    return eigenvalues