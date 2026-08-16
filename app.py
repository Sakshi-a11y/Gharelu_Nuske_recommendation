import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. DISEASE MODEL
# =====================================================

disease_model = joblib.load("disease_model.pkl")

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


SYMPTOM_KEYWORDS = {
    "fever": ["fever", "temperature"],
    "cough": ["cough", "coughing"],
    "headache": ["headache", "head pain", "head hurts"],
    "nausea": ["nausea"],
    "vomiting": ["vomiting", "vomit", "throwing up"],
    "fatigue": ["fatigue", "tired", "weakness", "weak"],
    "sore_throat": ["sore throat", "throat pain"],
    "chills": ["chills", "shivering"],
    "body_pain": ["body pain", "body ache", "body aches"],
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
    "diarrhea": [
        "diarrhea",
        "loose motion",
        "loose motions"
    ],
    "sweating": ["sweating", "excessive sweating"],
    "rapid_breathing": [
        "rapid breathing",
        "fast breathing"
    ],
    "dizziness": [
        "dizziness",
        "dizzy"
    ]
}


# =====================================================
# 2. REMEDY DATASET
# =====================================================

df = pd.read_csv("clean_remedies.csv")

df["Health Issue"] = df["Health Issue"].fillna("").astype(str)
df["Home Remedy"] = df["Home Remedy"].fillna("").astype(str)
df["Yogasan"] = df["Yogasan"].fillna("").astype(str)


# Remove URLs from Yogasan
df["Yogasan"] = df["Yogasan"].str.replace(
    r"https?://\S+",
    "",
    regex=True
).str.strip()


# Remove contaminated Dog-bite text
df["Home Remedy"] = df["Home Remedy"].str.replace(
    r"Dog-bite.*",
    "",
    regex=True,
    flags=__import__("re").IGNORECASE
).str.strip()


# Text used for embeddings
df["search_text"] = (
    "Health Issue: " + df["Health Issue"] +
    ". Home Remedy: " + df["Home Remedy"]
)


# =====================================================
# 3. FAISS
# =====================================================

print("Loading remedy embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    df["search_text"].tolist(),
    convert_to_numpy=True
)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

remedy_index = faiss.IndexFlatIP(dimension)

remedy_index.add(embeddings)

print("FAISS ready:", remedy_index.ntotal, "records")


# =====================================================
# 4. SYMPTOM EXTRACTION
# =====================================================

def extract_symptoms(text):

    text = text.lower()

    detected = []

    for symptom, keywords in SYMPTOM_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                detected.append(symptom)

                break

    return detected


# =====================================================
# 5. DISEASE PREDICTION
# =====================================================

def predict_disease(text):

    detected = extract_symptoms(text)

    symptoms = {}

    for feature in FEATURES:

        if feature in detected:
            symptoms[feature] = 1
        else:
            symptoms[feature] = 0

    input_data = pd.DataFrame(
        [[symptoms[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = disease_model.predict(
        input_data
    )[0]

    probabilities = disease_model.predict_proba(
        input_data
    )[0]

    confidence = max(probabilities)

    return detected, prediction, confidence


# =====================================================
# 6. REMEDY SEARCH
# =====================================================

def search_remedy(query, top_k=8):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = remedy_index.search(
        query_embedding,
        top_k
    )

    grouped = {}

    for score, idx in zip(scores[0], indices[0]):

        score = float(score)

        if score < 0.45:
            continue

        row = df.iloc[idx]

        issue = row["Health Issue"]

        if issue not in grouped:

            grouped[issue] = {
                "score": score,
                "remedies": [],
                "yogasan": []
            }

        remedy = row["Home Remedy"]

        if remedy and remedy not in grouped[issue]["remedies"]:

            grouped[issue]["remedies"].append(remedy)

        yoga = row["Yogasan"]

        if yoga and yoga not in grouped[issue]["yogasan"]:

            grouped[issue]["yogasan"].append(yoga)

    return grouped


# =====================================================
# 7. ROUTER
# =====================================================

def process_query(query):

    detected_symptoms = extract_symptoms(query)

    # 2 or more disease symptoms
    if len(detected_symptoms) >= 2:

        print("\n==============================")
        print("DISEASE ROUTE")
        print("==============================")

        detected, disease, confidence = predict_disease(
            query
        )

        print("\nDetected symptoms:")

        for symptom in detected:
            print("✓", symptom)

        print(
            "\nPredicted condition:",
            disease
        )

        print(
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )

    else:

        print("\n==============================")
        print("REMEDY ROUTE")
        print("==============================")

        results = search_remedy(query)

        if not results:

            print(
                "\nNo relevant home remedy found."
            )

            return

        for issue, data in results.items():

            print(
                "\nHealth Issue:",
                issue
            )

            print(
                "Similarity:",
                round(data["score"], 3)
            )

            print("\nHome Remedies:")

            for remedy in data["remedies"][:2]:

                print("-", remedy)

            if data["yogasan"]:

                print("\nYogasan:")

                for yoga in data["yogasan"][:1]:

                    print("-", yoga)


# =====================================================
# 8. CHAT LOOP
# =====================================================

print("\n================================")
print("HOME REMEDIES AI")
print("================================")

while True:

    query = input(
        "\nDescribe your problem (type 'exit' to stop): "
    )

    if query.lower() == "exit":
        print("Goodbye!")
        break

    process_query(query)