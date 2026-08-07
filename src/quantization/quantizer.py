import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq

class PowerOfTwoSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_exp, max_exp):
        # Prevent log2(0)
        x_safe = torch.where(x == 0, torch.tensor(1e-9, device=x.device), x)
        
        sign = torch.sign(x_safe)
        log2_val = torch.round(torch.log2(torch.abs(x_safe)))
        clamped_exp = torch.clamp(log2_val, min_exp, max_exp)
        
        return sign * (2.0 ** clamped_exp)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator passes gradient unmodified
        return grad_output, None, None

class PowerOfTwoFakeQuantize(nn.Module):
    def __init__(self, min_exp=-8, max_exp=1):
        super().__init__()
        self.min_exp = min_exp
        self.max_exp = max_exp

    def forward(self, x):
        return PowerOfTwoSTE.apply(x, self.min_exp, self.max_exp)

def get_asymmetric_activation_quantizer():
    # Asymmetric INT8 quantization (0 to 255)
    return tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )()

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias, padding_mode)
        
        self.weight_fake_quant = PowerOfTwoFakeQuantize(min_exp=-8, max_exp=1)
        self.act_fake_quant = get_asymmetric_activation_quantizer()

    def forward(self, input):
        q_weight = self.weight_fake_quant(self.weight)
        
        out = F.conv2d(input, q_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        
        out = self.act_fake_quant(out)
        return out

class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
        self.weight_fake_quant = PowerOfTwoFakeQuantize(min_exp=-8, max_exp=1)
        self.act_fake_quant = get_asymmetric_activation_quantizer()

    def forward(self, input):
        q_weight = self.weight_fake_quant(self.weight)
        
        out = F.linear(input, q_weight, self.bias)
        out = self.act_fake_quant(out)
        return out
    
def replace_layers_for_quantization(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            q_conv = QuantizedConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None)
            )
            
            # Copy original FP32 parameters
            q_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                q_conv.bias.data.copy_(child.bias.data)
                
            setattr(module, name, q_conv)
            
        elif isinstance(child, nn.Linear):
            q_linear = QuantizedLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None)
            )
            
            q_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                q_linear.bias.data.copy_(child.bias.data)
                
            setattr(module, name, q_linear)
            
        else:
            # Recursive call for sequential blocks (like ResNet layers)
            replace_layers_for_quantization(child)
            
def calibrate_ptq(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                  device: torch.device, num_batches: int = 10) -> None:
    model.eval()
    
    # Enable observers to collect activation statistics
    model.apply(torch.ao.quantization.enable_observer)
    model.apply(torch.ao.quantization.enable_fake_quant)
    
    batches_processed = 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            if batches_processed >= num_batches:
                break
                
            inputs = inputs.to(device)
            _ = model(inputs)
            
            batches_processed += 1
            
    # Disable observers, freeze calibration statistics
    model.apply(torch.ao.quantization.disable_observer)
    
def fuse_model_architectures(model: nn.Module, model_name: str) -> None:
    """
    Fuses Conv2d + BatchNorm2d to prevent QAT masking, 
    leaving the module as a standard nn.Conv2d for custom replacement.
    """
    model.eval() 
    
    if "resnet" in model_name:
        # Fuse stem without ReLU
        tq.fuse_modules(model, [['conv1', 'bn1']], inplace=True)
        
        for module_name, module_container in model.named_children():
            if module_name.startswith("layer"):
                for block in module_container:
                    # Fuse internal blocks without ReLU
                    tq.fuse_modules(block, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
                    
                    if hasattr(block, 'downsample') and block.downsample is not None:
                        tq.fuse_modules(block.downsample, [['0', '1']], inplace=True)
                        
    elif model_name == "cnn":
        # Fuse custom CNN without ReLU
        tq.fuse_modules(model, [
            ['conv1', 'bn1'],
            ['conv2', 'bn2'],
            ['conv3', 'bn3'],
            ['conv4', 'bn4']
        ], inplace=True)