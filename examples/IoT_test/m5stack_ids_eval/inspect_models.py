from pathlib import Path
import traceback
import torch


def summarize_tensor(v):
    shape = getattr(v, "shape", None)
    dtype = getattr(v, "dtype", None)
    return shape, dtype, type(v)


def inspect_state_dict(model_path: str, max_keys: int = 200):
    path = Path(model_path)
    print("=" * 100)
    print(f"INSPECTING: {path}")
    print("=" * 100)

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] torch.load failed for {path}")
        print(e)
        traceback.print_exc()
        return None

    print(f"TOP-LEVEL TYPE: {type(obj)}")

    if not isinstance(obj, dict):
        print("\nObject is not a dict. Full object:")
        print(obj)
        return obj

    print(f"\nNUM KEYS: {len(obj)}")
    print(f"\nFIRST {min(max_keys, len(obj))} KEYS:\n")

    for i, (k, v) in enumerate(obj.items()):
        shape, dtype, vtype = summarize_tensor(v)
        print(f"{i:03d} | {k} | shape={shape} | dtype={dtype} | type={vtype}")
        if i >= max_keys - 1:
            break

    return obj


def infer_layer_sizes_from_state_dict(state_dict):
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and hasattr(v, "shape") and len(v.shape) == 2:
            linear_weights.append((k, v.shape))

    linear_weights = sorted(linear_weights, key=lambda x: x[0])

    layer_sizes = []
    for i, (_, shape) in enumerate(linear_weights):
        out_features, in_features = shape
        if i == 0:
            layer_sizes.append(in_features)
        layer_sizes.append(out_features)

    return layer_sizes, linear_weights


def try_exportable_summary(model_path: str):
    path = Path(model_path)
    print("\n" + "-" * 100)
    print(f"EXPORT CHECK: {path}")
    print("-" * 100)

    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    if not isinstance(sd, dict):
        print("[FAIL] Model is not a dict state_dict.")
        return

    layer_sizes, linear_weights = infer_layer_sizes_from_state_dict(sd)

    if not linear_weights:
        print("[FAIL] No 2D '.weight' tensors found. This model is not directly exportable with the simple exporter.")
        return

    print("[OK] Found candidate linear layers:")
    for k, shape in linear_weights:
        print(f"  - {k}: {tuple(shape)}")

    print(f"\nInferred layer sizes: {layer_sizes}")

    bias_keys = [k for k, v in sd.items() if k.endswith(".bias") and hasattr(v, "shape") and len(v.shape) == 1]
    print(f"Bias tensors found: {len(bias_keys)}")
    for k in bias_keys[:20]:
        print(f"  - {k}")


def main():
    base = Path.cwd()

    qat_model = base / "qat_pytorch_model_exp6.pth"
    normal_model = base / "normal_pytorch_model_exp6.pth"

    print("\n[1] Inspecting QAT model\n")
    inspect_state_dict(qat_model, max_keys=200)
    try_exportable_summary(qat_model)

    print("\n\n[2] Inspecting NORMAL model\n")
    inspect_state_dict(normal_model, max_keys=100)
    try_exportable_summary(normal_model)


if __name__ == "__main__":
    main()