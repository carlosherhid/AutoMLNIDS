"""
Loading, on a machine without a GPU, a predictor that AutoGluon saved from a job that had one.
"""
import io

import torch


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)


def allow_cpu_loading():
    """AutoGluon pickles its PyTorch networks with the tensors on the device they were trained on, and torch.storage._load_from_bytes
    (what the pickle calls to restore them) has no map_location, so a predictor trained on a GPU node cannot be loaded on a CPU node:
    "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False". When no GPU is visible this sends
    those loads to the CPU; with a GPU nothing changes. Call it before TabularPredictor.load()."""
    if torch.cuda.is_available() or torch.storage._load_from_bytes is _load_from_bytes:
        return
    # it takes torch's own name, so that pickling a model afterwards still refers to torch.storage._load_from_bytes
    _load_from_bytes.__module__ = 'torch.storage'
    _load_from_bytes.__name__ = _load_from_bytes.__qualname__ = '_load_from_bytes'
    torch.storage._load_from_bytes = _load_from_bytes
