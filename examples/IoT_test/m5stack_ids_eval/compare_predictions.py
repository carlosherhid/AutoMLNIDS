import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac_preds", required=True)
    parser.add_argument("--m5_preds", required=True)
    args = parser.parse_args()

    mac = pd.read_csv(args.mac_preds)
    m5 = pd.read_csv(args.m5_preds)

    n = min(len(mac), len(m5))
    mac = mac.iloc[:n].copy()
    m5 = m5.iloc[:n].copy()

    merged = pd.DataFrame({
        "true": mac["true"],
        "pred_mac": mac["pred"],
        "pred_m5": m5["pred"],
        "latency_mac_us": mac["latency_us"],
        "latency_m5_us": m5["latency_us"],
    })

    merged["match"] = merged["pred_mac"] == merged["pred_m5"]

    print(merged)
    print("\nPrediction agreement:", merged["match"].mean())
    print("Matches:", merged["match"].sum(), "/", len(merged))

    merged.to_csv("comparison_mac_vs_m5.csv", index=False)

if __name__ == "__main__":
    main()