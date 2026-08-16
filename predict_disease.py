import pandas as pd
import joblib

# Load trained model
model = joblib.load("disease_model.pkl")

FEATURES = [
    "fever",
    "cough",
    "headache",
    "nausea",
    "vomiting",
    "fatigue",
    "sore_throat",
    "chills",
    "body_pain",
    "loss_of_appetite",
    "abdominal_pain",
    "diarrhea",
    "sweating",
    "rapid_breathing",
    "dizziness"
]

# Different ways user may describe a symptom
SYMPTOM_KEYWORDS = {
    "fever": ["fever", "temperature", "high temperature"],
    "cough": ["cough", "coughing"],
    "headache": ["headache", "head pain", "head hurts"],
    "nausea": ["nausea", "feeling nauseous"],
    "vomiting": ["vomiting", "vomit", "throwing up"],
    "fatigue": ["fatigue", "tired", "weakness", "weak"],
    "sore_throat": ["sore throat", "throat pain", "throat hurts"],
    "chills": ["chills", "shivering", "shivering chills"],
    "body_pain": ["body pain", "body ache", "body aches", "muscle pain"],
    "loss_of_appetite": [
        "loss of appetite",
        "no appetite",
        "not hungry"
    ],
    "abdominal_pain": [
        "abdominal pain",
        "stomach pain",
        "belly pain"
    ],
    "diarrhea": ["diarrhea", "loose motion", "loose motions"],
    "sweating": ["sweating", "excessive sweating"],
    "rapid_breathing": [
        "rapid breathing",
        "fast breathing",
        "breathing fast"
    ],
    "dizziness": ["dizziness", "dizzy", "feeling dizzy"]
}


def extract_symptoms(text):

    text = text.lower()

    symptoms = {}

    for feature in FEATURES:

        symptoms[feature] = 0

        for keyword in SYMPTOM_KEYWORDS[feature]:

            if keyword in text:
                symptoms[feature] = 1
                break

    return symptoms


def predict_disease(user_text):

    symptoms = extract_symptoms(user_text)

    # Create input in EXACT same feature order
    input_data = pd.DataFrame(
        [[symptoms[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    print("\nDetected Symptoms:")

    detected = False

    for feature in FEATURES:

        if symptoms[feature] == 1:
            print("✓", feature)
            detected = True

    if not detected:
        print("No known symptoms detected.")

    print("\nPredicted condition:", prediction)

    print("\nProbabilities:")

    for disease, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            disease,
            ":",
            round(probability * 100, 2),
            "%"
        )


# -----------------------------
# Test
# -----------------------------

user_text = input("\nDescribe your symptoms: ")

predict_disease(user_text)