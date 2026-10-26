import pandas as pd

df = pd.read_csv("data/test.csv")

X = df.drop(columns=["ID", "Label"])
y = df["Label"]

out = X.copy()
out["Label"] = y
out.to_csv("data/test_device.csv", index=False, header=False)

print("Generado: data/test_device.csv")
print("Shape:", out.shape)
