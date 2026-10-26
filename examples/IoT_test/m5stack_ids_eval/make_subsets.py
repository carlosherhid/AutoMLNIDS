import pandas as pd
from pathlib import Path

BASE = Path("data")
INPUT = BASE / "test_device.csv"

def main():
    df = pd.read_csv(INPUT, header=None)

    for n in [10, 100, 1000]:
        if len(df) >= n:
            out = BASE / f"test_device_{n}.csv"
            df.head(n).to_csv(out, index=False, header=False)
            print(f"Generated {out} with shape {df.head(n).shape}")

if __name__ == "__main__":
    main()