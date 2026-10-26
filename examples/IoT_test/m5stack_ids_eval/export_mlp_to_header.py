import argparse
import re
from pathlib import Path
import torch


def extract_layer_index(key: str):
    """
    Extract numeric layer index from keys like:
    main_block.1.weight
    main_block.13.bias
    """
    m = re.search(r"main_block\.(\d+)\.(weight|bias)$", key)
    if not m:
        return None
    return int(m.group(1))


def tensor_to_c_array(t, name):
    flat = t.detach().cpu().numpy().reshape(-1)
    values = ", ".join(f"{x:.8f}f" for x in flat)
    return f"static const float {name}[{len(flat)}] = {{{values}}};\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_header", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    if not isinstance(state_dict, dict):
        raise ValueError("The .pth file does not contain a standard state_dict.")

    # Collect linear layers in numeric order
    layer_entries = []
    for k, v in state_dict.items():
        idx = extract_layer_index(k)
        if idx is not None and k.endswith(".weight") and len(v.shape) == 2:
            layer_entries.append((idx, k, v))

    layer_entries.sort(key=lambda x: x[0])

    if not layer_entries:
        raise ValueError("No se encontraron capas lineales exportables.")

    # Infer layer sizes
    layer_sizes = []
    for i, (idx, k, w) in enumerate(layer_entries):
        out_features, in_features = w.shape
        if i == 0:
            layer_sizes.append(in_features)
        layer_sizes.append(out_features)

    # Biases in numeric order
    bias_entries = []
    for k, v in state_dict.items():
        idx = extract_layer_index(k)
        if idx is not None and k.endswith(".bias") and len(v.shape) == 1:
            bias_entries.append((idx, k, v))

    bias_entries.sort(key=lambda x: x[0])

    if len(bias_entries) != len(layer_entries):
        raise ValueError(
            f"Number of biases ({len(bias_entries)}) differs from the number of layers ({len(layer_entries)})."
        )

    lines = []
    lines.append("#pragma once\n")
    lines.append(f"// Auto-generated from {model_path.name}\n\n")
    lines.append(f"static const int {args.model_name}_num_layers = {len(layer_entries)};\n")
    lines.append(
        f"static const int {args.model_name}_layer_sizes[{len(layer_sizes)}] = "
        + "{"
        + ", ".join(str(x) for x in layer_sizes)
        + "};\n\n"
    )

    # Export weights
    for i, (_, k, w) in enumerate(layer_entries):
        lines.append(f"// {k} shape={tuple(w.shape)}\n")
        lines.append(tensor_to_c_array(w, f"{args.model_name}_w_{i}"))
        lines.append("\n")

    # Export biases
    for i, (_, k, b) in enumerate(bias_entries):
        lines.append(f"// {k} shape={tuple(b.shape)}\n")
        lines.append(tensor_to_c_array(b, f"{args.model_name}_b_{i}"))
        lines.append("\n")

    Path(args.output_header).write_text("".join(lines), encoding="utf-8")

    print(f"Header generado en: {args.output_header}")
    print(f"Num layers: {len(layer_entries)}")
    print(f"Layer sizes: {layer_sizes}")


if __name__ == "__main__":
    main()