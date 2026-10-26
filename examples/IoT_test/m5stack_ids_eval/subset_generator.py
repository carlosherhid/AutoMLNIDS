import pandas as pd
df = pd.read_csv("data/test_device.csv", header=None)
df.head(10).to_csv("data/test_device_10.csv", index=False, header=False)
print("OK")