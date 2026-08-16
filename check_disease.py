import pandas as pd

df = pd.read_csv("dataset.csv")

print("Total columns:", len(df.columns))
print("\nFeature columns:")

for i, col in enumerate(df.columns):
    if col != "label":
        print(i, ":", col)

print("\nTarget:")
print("label")