import torch

def calibrate_model(model, data_loader, num_batches=20, device="cpu"):
    """
    Feeds data through the model to allow Observers to record statistics 
    (min/max values of activations and weights).

    Args:
        model: The model with attached Observers (prepared model).
        data_loader: The data loader (usually training data).
        num_batches: How many batches to use for calibration.
        device: 'cpu' or 'cuda'. Quantization calibration is often done on CPU.
    """
    model.eval()
    model.to(device)
    
    # Counter to stop after num_batches
    cnt = 0
    
    # Disable gradient calculation (not needed for calibration)
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            model(inputs)
            
            cnt += 1
            if cnt >= num_batches:
                break
                
    print(f"Calibration finished using {num_batches} batches.")