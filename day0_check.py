"""
Day 0 check (GPU path): does torchao preserve PoT weight values, or overwrite them
with its own affine quantization?

The question: torchao's quantize_ applies torchao's OWN affine math. If we hand it
PoT-rounded weights, does the int8 tensor it produces still dequantize back to our
PoT values, or does torchao re-quantize with a different scale and destroy the PoT
structure?

Run this on ONE layer. It prints whether PoT survives conversion.

Requires: torch, torchao. Check version first:
    python -c "import torchao; print(torchao.__version__)"
"""

import torch
import torch.nn as nn
from src.quantization.quantizer import PowerOfTwoFakeQuantize

# -------------------------------------------------------------------
# 1. PoT quantization stand-in.
# Replace this with your actual PoT quantizer (PowerOfTwoSTE / weight_fake_quant)
# so the test uses your real scheme. This stand-in matches your logarithmic PoT-value
# method: sign * 2^round(log2|w|), clamped to an exponent range.
# -------------------------------------------------------------------
pot_quant = PowerOfTwoFakeQuantize(min_exp=-8, max_exp=1)

def pot_quantize(w: torch.Tensor) -> torch.Tensor:
    return pot_quant(w)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
 
    # a single linear layer as the test case (conv would behave the same way)
    layer = nn.Linear(512, 128, bias=False).to(device)
 
    # apply PoT to its weights: this is the tensor torchao must preserve
    with torch.no_grad():
        layer.weight.copy_(pot_quantize(layer.weight))
    pot_weight = layer.weight.detach().clone()
 
    # sanity: confirm the weights really are powers of two now
    unique_abs = torch.unique(pot_weight.abs())
    print("distinct |weight| values after PoT:", unique_abs.cpu().numpy())
    print("(should all be powers of two or zero)\n")
 
    # -------------------------------------------------------------------
    # 2. Apply torchao int8. torchao 0.17 uses the class-form config.
    #    Int8DynamicActivationInt8Weight = real int8 compute (dynamic act + int8 weight).
    # -------------------------------------------------------------------
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
 
    # keep an un-quantized reference layer with the SAME PoT weights for comparison
    ref_layer = nn.Linear(512, 128, bias=False).to(device)
    with torch.no_grad():
        ref_layer.weight.copy_(pot_weight)
 
    quantize_(layer, Int8DynamicActivationInt8WeightConfig())
 
    # -------------------------------------------------------------------
    # 3. Compare by FUNCTIONAL OUTPUT rather than tensor introspection.
    #    The quantized tensor subclass does not implement .dequantize(), but we can
    #    still ask the real question: does the int8 layer compute the same result as
    #    the PoT layer? Feed identical input through both and compare outputs.
    #    Output difference reflects the weight-encoding difference plus int8 rounding.
    # -------------------------------------------------------------------
    x = torch.randn(64, 512, device=device)
    with torch.no_grad():
        out_pot = ref_layer(x)          # PoT weights, full precision compute
        out_int8 = layer(x)             # torchao int8 layer
 
    diff = (out_int8 - out_pot).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    rel = diff.norm().item() / (out_pot.norm().item() + 1e-12)
 
    print("output max abs error  :", max_abs_err)
    print("output mean abs error :", mean_abs_err)
    print("output relative error :", rel)
    print()
 
    # verdict: does the int8 layer reproduce the PoT layer's output?
    if rel < 0.05:
        print("PoT PRESERVED (functionally): int8 output matches the PoT output.")
        print("  -> torchao's int8 encoding reproduces the PoT computation; GPU path viable.")
    elif rel < 0.20:
        print("PARTIAL: int8 output is close but not exact.")
        print("  -> torchao likely kept the structure but re-scaled. Inspect before trusting.")
        print("     Report these numbers; do not conclude either way yet.")
    else:
        print("PoT OVERWRITTEN: int8 output diverges from the PoT output.")
        print("  -> The stock config does not preserve PoT. Options:")
        print("     (a) use torchao low-level primitives to inject your own scales,")
        print("     (b) fall back to the CPU torch.ao.quantization path (Check B),")
        print("     (c) use the Phase-2 fallback (fake-quant accuracy + memory).")
 
 
if __name__ == "__main__":
    main()
