import pandas as pd
import joblib

# Load trained model
model = joblib.load("disease_model.pkl")

# Symptoms
symptoms = {
    "fever": 1,
    "cough": 1,
    "headache": 1,
    "nausea": 0,
    "vomiting": 0,
    "fatigue": 1,
    "sore_throat": 0,
    "chills": 1,
    "body_pain": 1,
    "loss_of_appetite": 1,
    "abdominal_pain": 1,
    "diarrhea": 0,
    "sweating": 0,
    "rapid_breathing": 0,
    "dizziness": 0
}

# Convert to DataFrame
input_data = pd.DataFrame([symptoms])

# Prediction
prediction = model.predict(input_data)

print("Predicted condition:", prediction[0])