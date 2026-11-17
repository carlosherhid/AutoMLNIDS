"""
Helpers for the scripts that convert or run INT8 models. This module only needs PyTorch (no AutoGluon), so it can be imported anywhere.
"""
import torch
import torch.nn as nn

# The engine that runs every reported INT8 number. The QAT configuration stays 'qnnpack' (uint8 activations, symmetric per-tensor int8 weights),
# which is the arithmetic the firmware implements (one requantization multiplier per layer); onednn executes it without the 16-bit saturation of
# the x86/fbgemm AVX2 kernels, and it is the engine against which the integer emulation of the firmware was checked.
DEFAULT_ENGINE = 'onednn'


def set_quantized_engine(name=DEFAULT_ENGINE):
    """Pin torch.backends.quantized.engine explicitly instead of inheriting PyTorch's default (x86), and return its name."""
    supported = torch.backends.quantized.supported_engines
    if name not in supported:
        raise SystemExit(f'quantized engine {name!r} is not available in this PyTorch build (supported: {supported})')
    torch.backends.quantized.engine = name
    return name


def fold_batchnorm_into_next_linear(model):
    """Fold, in place, every BatchNorm1d of an AutoGluon-style network into the Linear that follows it, and return how many were folded.

    AutoGluon places the normalization before the Linear (BN -> Dropout -> Linear), so Linear+BN fusion (`fuse_modules_qat`) does not apply; in
    inference the BN is an affine map per input feature, y = s * x + t with s = gamma / sqrt(var + eps) and t = beta - s * mean, which gives
    W' = W * s (column-wise) and b' = b + W t. Each folded BatchNorm1d is replaced by nn.Identity() so the layer indices (and therefore the
    state_dict keys) stay those of the BatchNorm-free replica. Only Dropout/Identity may sit between the BN and its Linear. Use it on a trained
    model: the running statistics are the ones that are folded."""
    sequence = model.main_block if hasattr(model, 'main_block') else model
    folded = 0
    for index in range(len(sequence)):
        layer = sequence[index]
        if not isinstance(layer, nn.BatchNorm1d):
            continue
        target = None
        for later in range(index + 1, len(sequence)):
            candidate = sequence[later]
            if isinstance(candidate, nn.Linear):
                target = candidate
                break
            if not isinstance(candidate, (nn.Dropout, nn.Identity)):
                raise ValueError(f'BatchNorm1d at {index} is followed by {type(candidate).__name__} before any Linear: it cannot be folded')
        if target is None:
            raise ValueError(f'BatchNorm1d at {index} has no Linear after it')
        if target.in_features != layer.num_features:
            raise ValueError(f'BatchNorm1d({layer.num_features}) at {index} does not match the next Linear ({target.in_features} inputs)')
        with torch.no_grad():
            scale = layer.weight.double() / torch.sqrt(layer.running_var.double() + layer.eps) if layer.affine else \
                1.0 / torch.sqrt(layer.running_var.double() + layer.eps)
            shift = (layer.bias.double() if layer.affine else 0.0) - scale * layer.running_mean.double()
            weight = target.weight.double()
            target.bias.copy_((target.bias.double() + weight @ shift).to(target.bias.dtype))
            target.weight.copy_((weight * scale.unsqueeze(0)).to(target.weight.dtype))
        sequence[index] = nn.Identity()
        folded += 1
    return folded
