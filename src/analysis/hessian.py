import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def compute_layerwise_hessian_trace(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 5,           # low to save time
    hutchinson_samples: int = 50    # Number of random v vectors
) -> dict:
    """
    Computes the layer-wise Hessian trace using Hutchinson's method via Double Backward.
    """
    model.eval()
    model.zero_grad()
    
    # Isolate only 2D/4D weight tensors (Conv2d and Linear layers)
    # We ignore biases and BatchNorm parameters to focus strictly on quantization targets
    target_params = {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and "weight" in name and param.dim() >= 2
    }
    
    
    if not target_params:
        logger.warning("No valid weight parameters found for Hessian analysis.")
        return {}
    
    traces = {name: 0.0 for name in target_params}
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        for name, param in target_params.items():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # first-order grad for this layer only
            grad = torch.autograd.grad(
                loss, param, create_graph=True, retain_graph=True
            )[0]

            for _ in range(hutchinson_samples):
                v = torch.randint_like(param, high=2, dtype=torch.float32) * 2 - 1.0
                v_dot_g = torch.sum(grad * v)

                hvp = torch.autograd.grad(
                    v_dot_g, param, retain_graph=True
                )[0]

                traces[name] += torch.sum(v * hvp).item() / (hutchinson_samples * num_batches)
            
            # Clear graphs to prevent massive Memory Leaks
            del outputs, loss, grad
            model.zero_grad()
            torch.cuda.empty_cache()

    return traces