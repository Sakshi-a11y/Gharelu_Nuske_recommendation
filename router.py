import pandas as pd
import joblib
import json
import os
from datetime import datetime

# =========================================================
# RAG + GEMINI
# =========================================================

from remedy_search import get_remedy_context
from gemini_service import generate_remedy_response


# =========================================================
# LOAD DISEASE MODEL
# =========================================================

model = joblib.load("disease_model.pkl")


# =========================================================
# CHAT HISTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")


# =========================================================
# DISEASE FEATURES
# =========================================================

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


# =========================================================
# CONFIDENCE THRESHOLD
# =========================================================

CONFIDENCE_THRESHOLD = 70


# =========================================================
# SYMPTOM KEYWORDS
# =========================================================

SYMPTOM_KEYWORDS = {

    "fever": [
        "fever",
        "temperature",
        "high temperature",
        "high fever"
    ],

    "cough": [
        "cough",
        "coughing"
    ],

    "headache": [
        "headache",
        "head pain",
        "head hurts",
        "head ache"
    ],

    "nausea": [
        "nausea",
        "feeling nauseous",
        "nauseous"
    ],

    "vomiting": [
        "vomiting",
        "vomit",
        "throwing up",
        "threw up",
        "throw up"
    ],

    "fatigue": [
        "fatigue",
        "tired",
        "weakness",
        "weak"
    ],

    "sore_throat": [
        "sore throat",
        "throat pain",
        "pain in throat"
    ],

    "chills": [
        "chills",
        "shivering",
        "shiver"
    ],

    "body_pain": [
        "body pain",
        "body ache",
        "body aches",
        "pain in body"
    ],

    "loss_of_appetite": [
        "loss of appetite",
        "no appetite",
        "not hungry",
        "loss appetite"
    ],

    "abdominal_pain": [
        "abdominal pain",
        "stomach pain",
        "belly pain",
        "stomach ache",
        "stomachache"
    ],

    "diarrhea": [
        "diarrhea",
        "diarrhoea",
        "loose motion",
        "loose motions",
        "loose stool",
        "loose stools"
    ],

    "sweating": [
        "sweating",
        "excessive sweating"
    ],

    "rapid_breathing": [
        "rapid breathing",
        "fast breathing",
        "breathing fast",
        "difficulty breathing"
    ],

    "dizziness": [
        "dizziness",
        "dizzy",
        "feeling dizzy"
    ]
}


# =========================================================
# EXTRACT SYMPTOMS
# =========================================================

def extract_symptoms(text):

    text = text.lower()

    detected = []

    for symptom, keywords in SYMPTOM_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                detected.append(symptom)

                break

    return detected


# =========================================================
# FOLLOW-UP QUESTIONS
# =========================================================

def get_follow_up_question(symptom):

    questions = {

        "fever":
            "How many days have you had fever?",

        "cough":
            "How many days have you had cough? "
            "Do you also have fever, sore throat, chills "
            "or difficulty breathing?",

        "headache":
            "How many days have you had headache? "
            "Do you also have fever, chills, nausea "
            "or body pain?",

        "nausea":
            "How many days have you had nausea? "
            "Do you also have vomiting, fever, abdominal "
            "pain or diarrhea?",

        "vomiting":
            "How many days have you been vomiting? "
            "Do you also have nausea, fever, abdominal "
            "pain or diarrhea?",

        "fatigue":
            "How many days have you been feeling weak "
            "or tired? Do you also have fever, chills, "
            "body pain or loss of appetite?",

        "sore_throat":
            "How many days have you had sore throat? "
            "Do you also have cough, fever or chills?",

        "chills":
            "How many days have you had chills? "
            "Do you also have fever, body pain, "
            "headache or fatigue?",

        "body_pain":
            "How many days have you had body pain? "
            "Do you also have fever, chills, headache, "
            "cough or fatigue?",

        "loss_of_appetite":
            "How many days have you had loss of appetite? "
            "Do you also have fever, nausea, vomiting "
            "or abdominal pain?",

        "abdominal_pain":
            "How many days have you had abdominal pain? "
            "Do you also have diarrhea, vomiting, nausea "
            "or fever?",

        "diarrhea":
            "How many days have you had diarrhea or loose "
            "motions? Do you also have abdominal pain, "
            "vomiting, nausea or fever?",

        "sweating":
            "How many days have you had excessive sweating? "
            "Do you also have fever, chills, fatigue "
            "or dizziness?",

        "rapid_breathing":
            "How long have you been experiencing rapid "
            "breathing? Are you also experiencing cough, "
            "fever or breathing difficulty?",

        "dizziness":
            "How many days have you had dizziness? "
            "Do you also have fever, headache, nausea "
            "or weakness?"
    }

    return questions.get(
        symptom,
        "Are you experiencing any other symptoms?"
    )


# =========================================================
# DISEASE PREDICTION
# =========================================================

def predict_disease(symptoms):

    input_data = {
        feature: 0
        for feature in FEATURES
    }

    for symptom in symptoms:

        if symptom in input_data:

            input_data[symptom] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    classes = model.classes_

    confidence = max(probabilities)

    return (
        prediction,
        confidence,
        classes,
        probabilities
    )


# =========================================================
# DISEASE SAFETY LAYER
# =========================================================

def disease_safety_check(
    prediction,
    confidence,
    symptoms,
    fever_duration=None
):

    confidence_percent = confidence * 100

    safety_flags = []

    # -----------------------------------------------------
    # LOW CONFIDENCE
    # -----------------------------------------------------

    if confidence_percent < CONFIDENCE_THRESHOLD:

        safety_flags.append(
            "LOW_CONFIDENCE"
        )

    # -----------------------------------------------------
    # FEVER FOR 5 OR MORE DAYS
    # -----------------------------------------------------

    if fever_duration is not None:

        try:

            # Handles values like:
            # 5
            # "5"
            # "5 days"

            duration_text = str(
                fever_duration
            )

            digits = ""

            for char in duration_text:

                if char.isdigit():
                    digits += char

                elif digits:
                    break

            if digits:

                days = int(digits)

                if days >= 5:

                    safety_flags.append(
                        "PERSISTENT_FEVER"
                    )

        except ValueError:

            pass

    # -----------------------------------------------------
    # VOMITING + DIZZINESS
    # -----------------------------------------------------

    if (
        "vomiting" in symptoms
        and
        "dizziness" in symptoms
    ):

        safety_flags.append(
            "VOMITING_DIZZINESS"
        )

    # -----------------------------------------------------
    # RAPID BREATHING
    # -----------------------------------------------------

    if "rapid_breathing" in symptoms:

        safety_flags.append(
            "RAPID_BREATHING"
        )

    # -----------------------------------------------------
    # FINAL SAFETY LEVEL
    # -----------------------------------------------------

    if (
        "RAPID_BREATHING" in safety_flags
        or
        "VOMITING_DIZZINESS" in safety_flags
    ):

        safety_level = "HIGH"

    elif (
        "PERSISTENT_FEVER" in safety_flags
        or
        "LOW_CONFIDENCE" in safety_flags
    ):

        safety_level = "CAUTION"

    else:

        safety_level = "NORMAL"

    return safety_level, safety_flags


# =========================================================
# SHOW REMEDIES USING RAG + GEMINI
# =========================================================

def show_remedies(query):

    print("\n==============================")
    print("REMEDY ROUTE")
    print("==============================")

    print(
        "\nSearching knowledge base for:",
        query
    )

    # =====================================================
    # FAISS RETRIEVAL
    # =====================================================

    context = get_remedy_context(query)

    # =====================================================
    # NO RESULT
    # =====================================================

    if not context:

        print(
            "\nNo suitable remedy information "
            "found in the knowledge base."
        )

        print(
            "\nPlease describe your problem in more detail."
        )

        return

    # =====================================================
    # GEMINI GENERATION
    # =====================================================

    try:

        response = generate_remedy_response(
            query,
            context
        )

        print(
            "\n===================================="
        )

        print(
            "AI HOME-CARE RESPONSE"
        )

        print(
            "===================================="
        )

        print(response)

    except Exception as e:

        print(
            "\n⚠️ Gemini response could not be generated."
        )

        print(
            "Error:",
            e
        )

        print(
            "\nRetrieved information "
            "from knowledge base:"
        )

        print(context)

    # =====================================================
    # DISCLAIMER
    # =====================================================

    print(
        "\n⚠️ Important:"
    )

    print(
        "This information is general home-care guidance "
        "from the knowledge base and is not a medical diagnosis."
    )

    print(
        "If symptoms are severe, persistent, or worsening, "
        "consult a qualified healthcare professional."
    )


# =========================================================
# SAVE CASE TO CHAT HISTORY
# =========================================================

def save_case(
    route,
    prediction=None,
    confidence=None,
    safety_level=None,
    safety_flags=None
):

    # Nothing to save
    if not original_query and not conversation_symptoms:

        return

    # -----------------------------------------------------
    # LOAD EXISTING HISTORY
    # -----------------------------------------------------

    history = []

    if os.path.exists(HISTORY_FILE):

        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history = json.load(file)

            if not isinstance(history, list):

                history = []

        except (
            json.JSONDecodeError,
            FileNotFoundError
        ):

            history = []

    # -----------------------------------------------------
    # CREATE CASE
    # -----------------------------------------------------

    case = {

        "date":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "original_query":
            original_query,

        "symptoms":
            sorted(
                list(conversation_symptoms)
            ),

        "fever_duration":
            fever_duration,

        "fever_pattern":
            fever_pattern,

        "route":
            route
    }

    # -----------------------------------------------------
    # DISEASE INFORMATION
    # -----------------------------------------------------

    if prediction is not None:

        case["prediction"] = str(
            prediction
        )

    if confidence is not None:

        case["confidence"] = round(
            float(confidence),
            2
        )

    if safety_level is not None:

        case["safety_level"] = safety_level

    if safety_flags:

        case["safety_flags"] = safety_flags

    # -----------------------------------------------------
    # APPEND TO HISTORY
    # -----------------------------------------------------

    history.append(case)

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\n✓ Current case saved to chat_history.json"
    )
    print(
        "Saved at:",
        HISTORY_FILE
    )


# =========================================================
# SHOW CHAT HISTORY
# =========================================================

def show_history():

    print(
        "\n===================================="
    )

    print(
        "PREVIOUS CHAT HISTORY"
    )

    print(
        "===================================="
    )

    if not os.path.exists(HISTORY_FILE):

        print(
            "\nNo previous history found."
        )

        return

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

    except (
        json.JSONDecodeError,
        FileNotFoundError
    ):

        history = []

    if not history:

        print(
            "\nNo previous cases saved."
        )

        return

    # -----------------------------------------------------
    # DISPLAY LAST 10 CASES
    # -----------------------------------------------------

    for number, case in enumerate(
        history[-10:],
        start=1
    ):

        print(
            f"\nCase {number}"
        )

        print(
            "Date:",
            case.get("date", "N/A")
        )

        print(
            "Query:",
            case.get(
                "original_query",
                "N/A"
            )
        )

        print(
            "Symptoms:",
            ", ".join(
                case.get(
                    "symptoms",
                    []
                )
            )
        )

        print(
            "Route:",
            case.get(
                "route",
                "N/A"
            )
        )

        if case.get("prediction"):

            print(
                "Prediction:",
                case.get("prediction")
            )

            print(
                "Confidence:",
                case.get("confidence"),
                "%"
            )

        print(
            "------------------------------------"
        )


# =========================================================
# RESET CASE
# =========================================================

def reset_case(
    route="UNKNOWN",
    prediction=None,
    confidence=None,
    safety_level=None,
    safety_flags=None
):

    global conversation_symptoms
    global waiting_for_followup
    global original_query
    global followup_step
    global fever_duration
    global fever_pattern

    # =====================================================
    # IMPORTANT:
    # SAVE FIRST
    # =====================================================

    save_case(
        route=route,
        prediction=prediction,
        confidence=confidence,
        safety_level=safety_level,
        safety_flags=safety_flags
    )

    # =====================================================
    # CLEAR CURRENT CASE
    # =====================================================

    conversation_symptoms.clear()

    waiting_for_followup = False

    original_query = ""

    followup_step = 0

    fever_duration = None

    fever_pattern = None


# =========================================================
# MAIN CHAT VARIABLES
# =========================================================

conversation_symptoms = set()

waiting_for_followup = False

original_query = ""

followup_step = 0

fever_duration = None

fever_pattern = None


# =========================================================
# MAIN CHAT
# =========================================================

while True:

    # =====================================================
    # USER INPUT
    # =====================================================

    user_input = input(
        "\nEnter your query (type 'exit' to stop): "
    ).strip()

    if not user_input:

        continue

    # =====================================================
    # EXIT
    # =====================================================

    if user_input.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break

    # =====================================================
    # SHOW HISTORY COMMAND
    # =====================================================

    if user_input.lower() in [
        "history",
        "show history",
        "previous chats",
        "previous history"
    ]:

        show_history()

        continue

    # =====================================================
    # FOLLOW-UP MODE
    # =====================================================

    if waiting_for_followup:

        answer = user_input.lower().strip()

        # =================================================
        # FEVER FLOW
        # =================================================

        if "fever" in conversation_symptoms:

            # =================================================
            # STEP 1 - FEVER DURATION
            # =================================================

            if followup_step == 1:

                fever_duration = answer

                print(
                    "\nFever duration recorded:",
                    fever_duration
                )

                print(
                    "\nBot: Is the fever mostly during "
                    "the day, mostly at night, or "
                    "throughout the day?"
                )

                followup_step = 2

                continue

            # =================================================
            # STEP 2 - FEVER PATTERN
            # =================================================

            elif followup_step == 2:

                fever_pattern = answer

                print(
                    "\nFever pattern recorded:",
                    fever_pattern
                )

                print(
                    "\nBot: Do you also have chills, "
                    "cough, headache, nausea, vomiting, "
                    "body pain or any other symptom?"
                )

                followup_step = 3

                continue

            # =================================================
            # STEP 3 - EXTRA SYMPTOMS
            # =================================================

            elif followup_step == 3:

                # ---------------------------------------------
                # NO ADDITIONAL SYMPTOMS
                # ---------------------------------------------

                if answer in [
                    "no",
                    "nope",
                    "nah",
                    "nahi",
                    "na",
                    "no i don't",
                    "no i dont"
                ]:

                    print(
                        "\nNo additional symptoms reported."
                    )

                    print(
                        "\nRoute: REMEDY"
                    )

                    show_remedies(
                        original_query
                    )

                    reset_case(
                        route="REMEDY"
                    )

                    continue

                # ---------------------------------------------
                # EXTRACT SYMPTOMS
                # ---------------------------------------------

                new_symptoms = extract_symptoms(
                    user_input
                )

                if not new_symptoms:

                    print(
                        "\nI couldn't detect a known symptom."
                    )

                    print(
                        "Please enter symptoms such as "
                        "chills, cough, headache, nausea, "
                        "vomiting, body pain, etc."
                    )

                    continue

                conversation_symptoms.update(
                    new_symptoms
                )

                symptoms = list(
                    conversation_symptoms
                )

                # ---------------------------------------------
                # 3+ SYMPTOMS = DISEASE
                # ---------------------------------------------

                if len(symptoms) > 2:

                    route = "DISEASE"

                    waiting_for_followup = False

                    followup_step = 0

                else:

                    print(
                        "\nCurrent symptoms:"
                    )

                    for symptom in symptoms:

                        print(
                            "✓",
                            symptom
                        )

                    print(
                        "\nBot: Please tell me another symptom."
                    )

                    continue

        # =================================================
        # NON-FEVER FOLLOW-UP
        # =================================================

        else:

            # ---------------------------------------------
            # NO ADDITIONAL SYMPTOMS
            # ---------------------------------------------

            if answer in [
                "no",
                "nope",
                "nah",
                "nahi",
                "na",
                "no i don't",
                "no i dont"
            ]:

                print(
                    "\nNo additional symptoms reported."
                )

                print(
                    "\nRoute: REMEDY"
                )

                show_remedies(
                    original_query
                )

                reset_case(
                    route="REMEDY"
                )

                continue

            # ---------------------------------------------
            # EXTRACT NEW SYMPTOMS
            # ---------------------------------------------

            new_symptoms = extract_symptoms(
                user_input
            )

            if not new_symptoms:

                print(
                    "\nI couldn't detect a known symptom."
                )

                print(
                    "Please enter symptoms such as "
                    "fever, cough, headache, chills, "
                    "body pain, nausea, vomiting, etc."
                )

                continue

            conversation_symptoms.update(
                new_symptoms
            )

            symptoms = list(
                conversation_symptoms
            )

            # ---------------------------------------------
            # 3+ SYMPTOMS = DISEASE
            # ---------------------------------------------

            if len(symptoms) > 2:

                route = "DISEASE"

                waiting_for_followup = False

                followup_step = 0

            else:

                route = "FOLLOW_UP"

                symptom = symptoms[0]

                print(
                    "\nBot:",
                    get_follow_up_question(
                        symptom
                    )
                )

                continue

    # =====================================================
    # NEW QUERY
    # =====================================================

    else:

        original_query = user_input

        new_symptoms = extract_symptoms(
            user_input
        )

        conversation_symptoms.update(
            new_symptoms
        )

        symptoms = list(
            conversation_symptoms
        )

        # =================================================
        # ROUTING LOGIC
        # =================================================

        if len(symptoms) > 2:

            route = "DISEASE"

        elif len(symptoms) > 0:

            route = "FOLLOW_UP"

        else:

            route = "REMEDY"

    # =====================================================
    # DISPLAY CURRENT CASE MEMORY
    # =====================================================

    print(
        "\n===================================="
    )

    print(
        "CURRENT CASE MEMORY"
    )

    print(
        "===================================="
    )

    print(
        "Original query:",
        original_query
    )

    print(
        "Symptoms:",
        ", ".join(
            symptoms
        ) if symptoms else "None"
    )

    if fever_duration is not None:

        print(
            "Fever duration:",
            fever_duration
        )

    if fever_pattern is not None:

        print(
            "Fever pattern:",
            fever_pattern
        )

    # =====================================================
    # CURRENT SYMPTOMS
    # =====================================================

    print(
        "\n===================================="
    )

    print(
        "CURRENT SYMPTOMS"
    )

    print(
        "===================================="
    )

    if symptoms:

        for symptom in symptoms:

            print(
                "✓",
                symptom
            )

    else:

        print(
            "None"
        )

    print(
        "\nRoute:",
        route
    )

    # =====================================================
    # REMEDY ROUTE
    # =====================================================

    if route == "REMEDY":

        show_remedies(
            original_query
        )

        reset_case(
            route="REMEDY"
        )

        continue

    # =====================================================
    # FOLLOW-UP ROUTE
    # =====================================================

    elif route == "FOLLOW_UP":

        symptom = symptoms[0]

        # ---------------------------------------------
        # FEVER
        # ---------------------------------------------

        if symptom == "fever":

            print(
                "\nBot: How many days have you had fever?"
            )

            followup_step = 1

        # ---------------------------------------------
        # OTHER SYMPTOMS
        # ---------------------------------------------

        else:

            print(
                "\nBot:",
                get_follow_up_question(
                    symptom
                )
            )

        waiting_for_followup = True

        continue

    # =====================================================
    # DISEASE ROUTE
    # =====================================================

    elif route == "DISEASE":

        print(
            "\n=============================="
        )

        print(
            "DISEASE ROUTE"
        )

        print(
            "=============================="
        )

        print(
            "\nSymptoms collected:"
        )

        for symptom in symptoms:

            print(
                "✓",
                symptom
            )

        # =================================================
        # RANDOM FOREST
        # =================================================

        prediction, confidence, classes, probabilities = \
            predict_disease(
                symptoms
            )

        confidence_percent = (
            confidence * 100
        )

        # =================================================
        # PREDICTION
        # =================================================

        print(
            "\nPredicted condition:",
            prediction
        )

        print(
            "Confidence:",
            round(
                confidence_percent,
                2
            ),
            "%"
        )

        # =================================================
        # PROBABILITIES
        # =================================================

        print(
            "\nProbabilities:"
        )

        for disease, probability in zip(
            classes,
            probabilities
        ):

            print(
                f"{disease} : "
                f"{round(probability * 100, 2)} %"
            )

        # =================================================
        # SAFETY CHECK
        # =================================================

        safety_level, safety_flags = \
            disease_safety_check(
                prediction=prediction,
                confidence=confidence,
                symptoms=symptoms,
                fever_duration=fever_duration
            )

        print(
            "\nSafety Level:",
            safety_level
        )

        # =================================================
        # SAFETY FLAGS
        # =================================================

        if safety_flags:

            print(
                "\nSafety Flags:"
            )

            for flag in safety_flags:

                print(
                    "⚠",
                    flag
                )

        # =================================================
        # HIGH SAFETY LEVEL
        # =================================================

        if safety_level == "HIGH":

            print(
                "\n⚠️ Important:"
            )

            print(
                "Some symptoms require prompt "
                "medical evaluation."
            )

            print(
                "Please consult a qualified "
                "healthcare professional."
            )

        # =================================================
        # LOW CONFIDENCE
        # =================================================

        elif confidence_percent < CONFIDENCE_THRESHOLD:

            print(
                "\n⚠️ Low confidence prediction."
            )

            print(
                "The available symptoms are not "
                "sufficient for a reliable prediction."
            )

            print(
                "Please consult a qualified doctor "
                "for proper evaluation."
            )

        # =================================================
        # NORMAL / CAUTION
        # =================================================

        else:

            print(
                "\nPossible condition:",
                prediction
            )

            print(
                "\n⚠️ Important:"
            )

            print(
                "This is an AI-based prediction, "
                "not a medical diagnosis."
            )

            print(
                "Please consult a qualified healthcare "
                "professional for proper diagnosis."
            )

        # =================================================
        # SAVE + CLEAR CURRENT CASE
        # =================================================

        reset_case(
            route="DISEASE",
            prediction=prediction,
            confidence=confidence_percent,
            safety_level=safety_level,
            safety_flags=safety_flags
        )

        print(
            "\nPrevious case cleared."
        )

        print(
            "Next query will be treated "
            "as a NEW case."
        )