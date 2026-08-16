import pandas as pd

df = pd.read_csv("home_remedies.csv")

print("Original shape:", df.shape)

# -----------------------------
# 1. Missing values
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 2. Exact duplicate rows
# -----------------------------
print("\nExact duplicate rows:", df.duplicated().sum())

# -----------------------------
# 3. Duplicate health issues
# -----------------------------
print("\nHealth issues with multiple remedies:")
counts = df["Health Issue"].value_counts()

print(counts[counts > 1])

# -----------------------------
# 4. Empty / whitespace values
# -----------------------------
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

print("\nEmpty values after cleaning:")
print((df == "").sum())

# -----------------------------
# 5. Show all Indigestion rows
# -----------------------------
print("\nIndigestion entries:")
print(
    df[df["Health Issue"].str.lower() == "indigestion"][
        ["Health Issue", "Home Remedy", "Yogasan"]
    ].to_string(index=False)
)

# -----------------------------
# 6. Show all Gas-related entries
# -----------------------------
keywords = ["gas", "indigestion", "flatulence", "acidity"]

gas_df = df[
    df["Health Issue"]
    .str.lower()
    .str.contains("|".join(keywords), na=False)
]

print("\nGas/digestion related entries:")
print(
    gas_df[
        ["Health Issue", "Home Remedy", "Yogasan"]
    ].to_string(index=False)
)