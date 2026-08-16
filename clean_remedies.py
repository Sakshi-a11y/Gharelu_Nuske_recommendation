import pandas as pd

# Load dataset
df = pd.read_csv("home_remedies.csv")

print("Original shape:", df.shape)

# Keep only useful columns
df = df[["Health Issue", "Home Remedy", "Yogasan"]].copy()

# Remove extra spaces
for col in df.columns:
    df[col] = df[col].astype("string").str.strip()

# Convert empty strings to missing values
df = df.replace("", pd.NA)

# Remove rows where Home Remedy is missing
df = df.dropna(subset=["Health Issue", "Home Remedy"])

# Remove exact duplicate combinations
df = df.drop_duplicates(
    subset=["Health Issue", "Home Remedy", "Yogasan"]
)

# Save cleaned dataset
df.to_csv("clean_remedies.csv", index=False)

print("\nCleaned shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved as: clean_remedies.csv")