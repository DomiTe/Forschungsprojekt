import sys
from pathlib import Path

import torch

# make the project root importable when running from notebooks/
PROJECT_ROOT = Path.cwd().parent  # adjust if your notebook lives elsewhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
print('project root:', PROJECT_ROOT)

from src.utility.config import DATASET_SPECS, BASE_DIR

MODEL_NAME = 'cnn'
DATASET_NAME = 'IMAGENET100'
RUN_ID = '20260708_140413_24210'  # run that produced the QAT models

# QAT models are saved as qat_po2_{model}_{dataset}.pt under quantized_models/
qat_path = (Path(BASE_DIR) / 'results' / RUN_ID / 'quantized_models'
            / f'qat_po2_{MODEL_NAME}_{DATASET_NAME}.pt')
print('looking for:', qat_path)
print('exists:', qat_path.exists())

# Cell 3: the diagnostic functions
def analyze_pot_dynamic_range(pot_weight: torch.Tensor) -> dict:
    """Per output channel (ch_axis=0), measure PoT dynamic range and distinct values."""
    out_channels = pot_weight.shape[0]
    flat = pot_weight.reshape(out_channels, -1)
    results = {'dynamic_range': [], 'num_distinct': []}

    for c in range(out_channels):
        abs_w = flat[c].abs()
        nonzero = abs_w[abs_w > 0]
        if nonzero.numel() == 0:
            results['dynamic_range'].append(0.0)
            results['num_distinct'].append(0)
            continue
        results['dynamic_range'].append((nonzero.max() / nonzero.min()).item())
        results['num_distinct'].append(torch.unique(nonzero).numel())
    return results


def summarize_pot_representability(model, int8_levels: int = 256) -> dict:
    """Fraction of channels whose PoT dynamic range fits a uniform int8 grid."""
    from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear

    all_ranges = []
    dead_channels = 0
    per_layer = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            pot_weight = module.weight_fake_quant(module.weight)
            stats = analyze_pot_dynamic_range(pot_weight)
            all_ranges.extend(stats['dynamic_range'])
            dead_channels += stats['num_distinct'].count(0)
            per_layer[name] = stats

    if not all_ranges:
        print('WARNING: no quantized conv/linear layers found. Is the model quantized?')
        return {'fraction_representable': 0.0, 'worst_case_range': 0.0,
                'total_channels': 0, 'dead_channels': 0, 'per_layer': {}}

    ranges_tensor = torch.tensor(all_ranges)
    return {
        'fraction_representable': (ranges_tensor <= int8_levels).float().mean().item(),
        'worst_case_range': ranges_tensor.max().item(),
        'total_channels': len(all_ranges),
        'dead_channels': dead_channels,
        'per_layer': per_layer,
    }

from src.model_cnn.train import build_model
from src.quantization.quantizer import (
    fuse_model_architectures,
    replace_layers_for_quantization,
    QuantizedConv2d,
    QuantizedLinear,
)

specs = DATASET_SPECS[DATASET_NAME]
qat_model = build_model(
    num_classes=specs['num_classes'],
    model_name=MODEL_NAME,
    channels=specs['channels'],
    image_size=specs['image_size'],
)

# rebuild the wrapped structure in the same order as the training pipeline
qat_model.eval()
fuse_model_architectures(qat_model, MODEL_NAME)
replace_layers_for_quantization(qat_model)
qat_model = qat_model.to(device)

state = torch.load(qat_path, map_location=device, weights_only=True)
qat_model.load_state_dict(state)
qat_model.eval()

# sanity check: confirm the quantizer modules are present after reload
n_quant = sum(isinstance(m, (QuantizedConv2d, QuantizedLinear))
              for m in qat_model.modules())
print(f'quantized layers found in loaded model: {n_quant}')
if n_quant == 0:
    print('  -> no quantized layers; check fuse/replace ran correctly.')

result = summarize_pot_representability(qat_model)

print(f"total channels        : {result['total_channels']}")
print(f"dead (all-zero) chans : {result['dead_channels']}")
print(f"fraction representable: {result['fraction_representable']:.1%}")
print(f"worst-case range      : {result['worst_case_range']:.1f}")
print()
if result['fraction_representable'] > 0.95 and result['worst_case_range'] < 256:
    print('Path 2 looks essentially lossless: PoT deploys cleanly to per-channel int8.')
elif result['fraction_representable'] > 0.7:
    print('Path 2 partially fits: some channels lose precision. Quantify the cost.')
else:
    print('Path 2 struggles: many channels exceed int8 range. Reconsider grouping.')

rows = []
for name, stats in result['per_layer'].items():
    ranges = [r for r in stats['dynamic_range'] if r > 0]
    if not ranges:
        continue
    rows.append((name, max(ranges), sum(ranges) / len(ranges), len(ranges)))

rows.sort(key=lambda r: r[1], reverse=True)
print(f"{'layer':<28}{'max_range':>12}{'mean_range':>12}{'channels':>10}")
print('-' * 62)
for name, mx, mean, n in rows:
    print(f'{name:<28}{mx:>12.1f}{mean:>12.1f}{n:>10}')
