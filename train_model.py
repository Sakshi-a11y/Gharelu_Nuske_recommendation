import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset.csv")
print("Columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nLabel values:")
print(df["label"].value_counts(dropna=False))

print("\nMissing labels:")
print(df["label"].isna().sum())

print("Dataset shape:", df.shape)

print("\nLabels:")
print(df["label"].value_counts())

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "disease_model.pkl")

print("\nModel saved as disease_model.pkl")