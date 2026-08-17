🩺 AI Home Remedies & Disease Prediction Chatbot

An AI-powered health assistant that combines symptom-based disease prediction, FAISS-based semantic retrieval, RAG, Gemini, and a safety layer to provide general home-care information and possible-condition predictions.


📌 Overview

This is a hybrid AI health assistant rather than a simple chatbot.

It uses:

Rule-based routing: REMEDY, FOLLOW_UP, DISEASE

Random Forest for symptom-based disease prediction

FAISS + Sentence Transformers for semantic remedy retrieval

RAG to ground remedy responses in the project knowledge base

Gemini to generate natural-language home-care responses

Confidence and safety checks

Case memory stored in chat_history.json

Ingredient image generation for remedies when a useful ingredient image/reference is unavailable

Architecture

User Query
    │
    ▼
  Router
    │
    ├── REMEDY
    │      └── FAISS → RAG Context → Gemini
    │
    ├── FOLLOW_UP
    │      └── Collect additional symptoms
    │
    └── DISEASE
           └── Random Forest → Confidence → Safety Layer

✨ Key Features

1. Intelligent Query Routing

The chatbot routes queries into:

REMEDY

FOLLOW_UP

DISEASE

Example:

"I have gas and indigestion" → REMEDY
"I have fever" → FOLLOW_UP

2. Follow-Up Symptom Collection

For fever-related cases, the system can collect:

Fever duration

Fever pattern

Chills

Cough

Headache

Nausea

Vomiting

Body pain

Other supported symptoms

3. Disease Prediction

The disease component uses a Random Forest Classifier.

Current supported conditions:

Malaria

Pneumonia

Typhoid

Current configuration:

Algorithm: Random Forest Classifier
Estimators: 100
Random State: 42
Train/Test Split: 80/20
Stratified Split: Yes

Current evaluation:

Accuracy: 86.7%

The model is probabilistic and must not be treated as a confirmed diagnosis.

🔎 FAISS-Based Remedy Retrieval

The remedy component uses FAISS (Facebook AI Similarity Search) with the Sentence Transformers model:

all-MiniLM-L6-v2

Pipeline:

User Query
    ↓
Sentence Transformer
    ↓
Embedding
    ↓
FAISS Similarity Search
    ↓
Relevant Remedy Records

The current remedy data contains fields such as:

Health Issue

Home Remedy

Yogasan

📚 RAG Pipeline

FAISS provides the retrieval component of the RAG system:

User Query
     ↓
Embedding Model
     ↓
FAISS Retrieval
     ↓
Relevant Knowledge
     ↓
RAG Context
     ↓
Gemini
     ↓
Final AI Response

Gemini receives the retrieved context and converts it into a readable response.

The prompt instructs Gemini to:

Use retrieved knowledge

Avoid unsupported remedies

Explain information simply

Mention incomplete knowledge

Avoid presenting responses as diagnoses

Recommend professional medical evaluation when appropriate

🤖 Gemini Integration

Gemini is used as the generation and explanation layer after retrieval.

Example:

FAISS retrieves remedy information
             ↓
           Gemini
             ↓
Structured home-care response

The Gemini API key is supplied through an environment variable and should never be committed to GitHub.

🖼️ Ingredient Image Generation

A planned AI feature will provide illustrative ingredient images when the remedy knowledge base does not contain a useful image/reference for an ingredient.

Example flow:

Remedy Retrieved
      ↓
Ingredient identified
      ↓
Image/reference unavailable
      ↓
AI image generation
      ↓
Ingredient illustration shown to user

This can make remedy instructions easier to understand visually.

Generated images are illustrative and should not be treated as authoritative identification of medicinal plants, herbs, or other ingredients.

🛡️ Confidence & Safety Layer

The disease route includes a safety layer.

Example:

Predicted condition: Malaria
Confidence: 93.0%

Safety Level: NORMAL

For persistent fever:

Safety Level: CAUTION

Safety Flags:
⚠ PERSISTENT_FEVER

For uncertain predictions:

⚠️ LOW CONFIDENCE

The available symptoms are not sufficient
for a reliable prediction.

🧠 Case Memory

The current case is temporarily maintained while follow-up questions are being answered.

Example:

Original query: I have fever
Symptoms: fever, cough, chills
Fever duration: 3 days
Fever pattern: night

Completed cases are stored in:

chat_history.json

Example record:

{
  "date": "2026-08-16 18:48:27",
  "original_query": "i have fever",
  "symptoms": ["chills", "cough", "fever"],
  "fever_duration": "3",
  "fever_pattern": "night",
  "route": "DISEASE",
  "prediction": "Malaria",
  "confidence": 93.0,
  "safety_level": "NORMAL"
}

The JSON history is currently a prototype. User-specific persistent storage is planned for the database phase.

🏗️ Current Architecture

                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Router.py       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          REMEDY           FOLLOW_UP         DISEASE
              │                │                │
              ▼                ▼                ▼
          FAISS/RAG      Symptom Memory    Random Forest
              │                                  │
              ▼                                  ▼
           Gemini                           Confidence
                                                 │
                                                 ▼
                                           Safety Layer
                                                 │
                                                 ▼
                                           Case Memory
                                                 │
                                                 ▼
                                        chat_history.json

📂 Project Structure

Current prototype:

home-remedies/
│
├── router.py
├── remedy_search.py
├── gemini_service.py
├── disease prediction files
│
├── disease_model.pkl
├── clean_remedies.csv
├── home_remedies.csv
├── disease dataset files
├── chat_history.json
├── requirements.txt
│
└── AI_project/
    └── Spring Boot backend

File names may differ slightly depending on the current local version.

⚙️ Technologies Used

AI / Machine Learning

Python

Pandas

Scikit-learn

Random Forest

Sentence Transformers

FAISS

RAG

Gemini API

Backend — In Progress

Java

Spring Boot

Spring Web

Maven

Frontend — In Progress

HTML

CSS

JavaScript

Database — Planned

MySQL

Future Integration

Google Maps / Places API

User authentication

User-specific chat history

🚀 Setup

1. Clone the repository

git clone (https://github.com/Sakshi-a11y/Gharelu_Nuske_recommendation.git)
cd home-remedies

2. Create a Python virtual environment

Windows:

python -m venv venv
.\venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

If you have not created requirements.txt yet:

pip freeze > requirements.txt

🔐 Gemini API Key

PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"

Then run the application in the same terminal.

For a permanent user-level variable:

[System.Environment]::SetEnvironmentVariable(
    "GEMINI_API_KEY",
    "YOUR_API_KEY",
    "User"
)

Restart the terminal after setting it permanently.

Never commit API keys, .env files, credentials, or private user data to GitHub.

▶️ Run the Current AI Prototype

python router.py

Example:

Enter your query (type 'exit' to stop): I have gas and indigestion

For a disease case:

Enter your query (type 'exit' to stop): I have fever

The router asks follow-up questions and then runs prediction and safety checks.

🧪 Example Disease Flow

User:
I have fever
        ↓
FOLLOW_UP
        ↓
How many days?
        ↓
3
        ↓
Fever pattern?
        ↓
night
        ↓
Additional symptoms?
        ↓
cough, chills
        ↓
DISEASE
        ↓
Prediction: Malaria
Confidence: 93%
Safety: NORMAL
        ↓
Case saved to chat_history.json

🧪 Example Remedy Flow

User:
I have gas and indigestion
        ↓
REMEDY
        ↓
FAISS Retrieval
        ↓
RAG Context
        ↓
Gemini
        ↓
AI Home-Care Response

🔮 Future Enhancements

1. Web UI

Build the chatbot interface using:

HTML + CSS + JavaScript

2. Spring Boot Backend

Target architecture:

Frontend
   ↓
Spring Boot REST API
   ↓
Python AI Service
   ↓
FAISS / Random Forest / Gemini

3. User Authentication

Add:

Registration

Login

Password hashing

Authentication/session management

4. MySQL Database

Move from JSON-based history to user-specific persistent storage.

Potential tables:

users
chat_history
health_cases

5. 🏥 Hospital Recommendation

Integrate Google Maps / Places services to recommend nearby hospitals or healthcare facilities when the safety layer indicates that medical evaluation may be appropriate.

Disease Prediction
      ↓
Safety Layer
      ↓
Medical Evaluation Recommended
      ↓
User Location
      ↓
Nearby Hospitals

The feature can provide useful information such as:

Nearby hospital/healthcare facility names

Distance/location

Map navigation

Available place information

The recommendation is intended to help users find healthcare facilities and does not determine which hospital the user medically needs.

6. 🖼️ Ingredient Image Generation

If the remedy knowledge base does not contain a useful image/reference for an ingredient, an AI image-generation feature can create an illustrative representation of that ingredient.

Remedy
  ↓
Ingredient
  ↓
Image/reference unavailable
  ↓
AI Image Generation
  ↓
Visual ingredient reference

This can make the chatbot more interactive and easier to understand.

Generated images are illustrative only and should not be used as definitive botanical or medicinal identification.

7. Better Conversational Memory

Use previous conversations to provide context while keeping each user's data separated by account.

8. Improved Dataset

Future versions can add:

More symptoms

More health conditions

Remedy metadata

Contraindication information

Source/reference information

📈 Current Project Status

Component

Status

Query Router

✅ Completed

Follow-up Symptom Collection

✅ Completed

Fever Duration & Pattern

✅ Completed

Random Forest Prediction

✅ Completed

Prediction Confidence

✅ Completed

Safety Layer

✅ Completed

FAISS Remedy Retrieval

✅ Completed

RAG Context

✅ Completed

Gemini Integration

✅ Completed

Case Memory

✅ Completed

JSON Chat History

✅ Completed

HTML/CSS/JS UI

🔄 In Progress

Spring Boot Backend

🔄 In Progress

MySQL Database

🔜 Planned

User Login

🔜 Planned

Hospital Recommendation

🔜 Planned

Ingredient Image Generation

🔜 Planned

🎯 Project Goal

The goal is to build a hybrid AI health assistant combining:

ML prediction + semantic retrieval + RAG + LLM generation + safety checks + conversational memory

instead of relying on a single LLM for every health-related task.

👩‍💻 Author

Sakshi Chitransh

B.Tech — Artificial Intelligence & Machine Learning

⚠️ Medical Safety

This project is an educational AI prototype.

It should not be used to:

Confirm a disease

Replace a doctor

Prescribe medication

Determine emergency treatment

Replace professional medical evaluation

The model predictions depend on the symptoms supplied to the system. The safety layer is designed to communicate uncertainty and encourage professional evaluation when appropriate.

📄 License

This project is intended for educational and development purposes. Add an appropriate open-source license if you plan to distribute or accept contributions.
