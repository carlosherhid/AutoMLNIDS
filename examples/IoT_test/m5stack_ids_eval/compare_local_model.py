import re
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


class CompactMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.main_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.main_block(x)


def extract_linear_indices(state_dict):
    idxs = []
    for k in state_dict.keys():
        m = re.match(r"main_block\.(\d+)\.weight", k)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def remap_state_dict_to_compact(state_dict):
    """
    Maps:
      main_block.1.weight  -> main_block.0.weight
      main_block.1.bias    -> main_block.0.bias
      main_block.5.weight  -> main_block.2.weight
      ...
    so it matches a compact Sequential:
      [Linear, ReLU, Linear, ReLU, ..., Linear]
    """
    linear_indices = extract_linear_indices(state_dict)

    remapped = {}
    compact_linear_idx = 0

    for orig_idx in linear_indices:
        compact_idx = compact_linear_idx * 2  # linear layers at 0,2,4,...
        w_key_old = f"main_block.{orig_idx}.weight"
        b_key_old = f"main_block.{orig_idx}.bias"
        w_key_new = f"main_block.{compact_idx}.weight"
        b_key_new = f"main_block.{compact_idx}.bias"

        remapped[w_key_new] = state_dict[w_key_old]
        remapped[b_key_new] = state_dict[b_key_old]

        compact_linear_idx += 1

    return remapped


def infer_layer_sizes_from_state_dict(state_dict):
    linear_indices = extract_linear_indices(state_dict)
    layer_sizes = []

    for i, orig_idx in enumerate(linear_indices):
        w = state_dict[f"main_block.{orig_idx}.weight"]
        out_features, in_features = w.shape
        if i == 0:
            layer_sizes.append(in_features)
        layer_sizes.append(out_features)

    return layer_sizes


def load_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    layer_sizes = infer_layer_sizes_from_state_dict(state_dict)

    model = CompactMLP(layer_sizes)
    remapped_state_dict = remap_state_dict_to_compact(state_dict)
    model.load_state_dict(remapped_state_dict, strict=True)
    model.eval()
    return model


def main():
    model = load_model("normal_pytorch_model_exp6.pth")

    df = pd.read_csv("data/test_device_10.csv", header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)

    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(x_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    out = pd.DataFrame({
        "true": y,
        "pred_python": preds
    })

    print(out)
    print("\nAccuracy Python local:", (out["true"] == out["pred_python"]).mean())
    out.to_csv("preds_python_exp6_normal_10.csv", index=False)


if __name__ == "__main__":
    main()