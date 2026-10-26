import argparse
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
    linear_indices = extract_linear_indices(state_dict)
    remapped = {}
    compact_linear_idx = 0

    for orig_idx in linear_indices:
        compact_idx = compact_linear_idx * 2
        remapped[f"main_block.{compact_idx}.weight"] = state_dict[f"main_block.{orig_idx}.weight"]
        remapped[f"main_block.{compact_idx}.bias"] = state_dict[f"main_block.{orig_idx}.bias"]
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
    model.load_state_dict(remap_state_dict_to_compact(state_dict), strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    model = load_model(args.model_path)

    df = pd.read_csv(args.input_csv, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)

    preds = []
    latencies_us = []

    with torch.no_grad():
        for row in X:
            x = torch.tensor(row, dtype=torch.float32).unsqueeze(0)

            t0 = time.perf_counter()
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            t1 = time.perf_counter()

            preds.append(pred)
            latencies_us.append((t1 - t0) * 1e6)

    out = pd.DataFrame({
        "pred": preds,
        "true": y,
        "latency_us": latencies_us
    })
    out.to_csv(args.output_csv, index=False)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("\nConfusion matrix:")
    print(cm)

    lat = np.array(latencies_us)
    print("\nLatency (us):")
    print("mean  :", np.mean(lat))
    print("median:", np.median(lat))
    print("p95   :", np.percentile(lat, 95))
    print("p99   :", np.percentile(lat, 99))
    print("min   :", np.min(lat))
    print("max   :", np.max(lat))


if __name__ == "__main__":
    main()