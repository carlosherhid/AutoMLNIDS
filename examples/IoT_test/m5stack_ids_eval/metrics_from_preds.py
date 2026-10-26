import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.preds)

    # discard invalid rows, if any
    df = df[(df["pred"] >= 0) & (df["true"] >= 0) & (df["latency_us"] >= 0)]

    y_true = df["true"].values
    y_pred = df["pred"].values
    lat = df["latency_us"].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("\nConfusion matrix:")
    print(cm)

    print("\nLatency (us):")
    print("mean  :", np.mean(lat))
    print("median:", np.median(lat))
    print("p95   :", np.percentile(lat, 95))
    print("p99   :", np.percentile(lat, 99))
    print("min   :", np.min(lat))
    print("max   :", np.max(lat))

if __name__ == "__main__":
    main()
