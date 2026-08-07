import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.utility.utils import TimingTracker

def train_qat(
    ptq_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4
) -> tuple[nn.Module, dict, TimingTracker]:
    
    qat_model = copy.deepcopy(ptq_model).to(device)
    qat_model.train()
    
    # Enable fake quant (STE) and observers
    qat_model.apply(torch.ao.quantization.enable_fake_quant)
    qat_model.apply(torch.ao.quantization.enable_observer)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, qat_model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4
    )
    
    # Cosine annealing decaying to 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    timer = TimingTracker()
    
    best_acc = 0.0
    best_state = None
    
    # Calculate when to freeze observers (e.g., epoch 8 out of 10)
    freeze_epoch = max(1, min(epochs, int(epochs * 0.8)))

    for epoch in range(1, epochs + 1):
        timer.start_epoch()
        
        # Freeze observers near the end to stabilize calibration scales
        if epoch == freeze_epoch:
            qat_model.apply(torch.ao.quantization.disable_observer)
            # Fake quant remains enabled so gradients continue to flow
            
        # --- Training phase ---
        timer.start_split("train")
        qat_model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        timer.end_split()

        # --- Validation phase ---
        timer.start_split("val")
        qat_model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = qat_model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        timer.end_split()
        timer.end_epoch(epoch)
        
        # Step the scheduler
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(qat_model.state_dict())

    if best_state is not None:
        qat_model.load_state_dict(best_state)
        
    return qat_model, history, timer