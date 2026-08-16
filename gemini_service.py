import os
from google import genai


# ==========================================
# GEMINI CLIENT
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


# ==========================================
# GENERATE RESPONSE USING RAG CONTEXT
# ==========================================

def generate_remedy_response(user_query, retrieved_context):

    prompt = f"""
You are a helpful home-care assistant.

USER QUERY:
{user_query}

INFORMATION RETRIEVED FROM OUR HOME-REMEDY KNOWLEDGE BASE:
{retrieved_context}

INSTRUCTIONS:

1. Use the retrieved information as the primary source.
2. Do not invent home remedies that are not present in the retrieved information.
3. Explain the available information in simple language.
4. Mention relevant precautions when available.
5. Do not claim that the user definitely has a disease.
6. If the retrieved information is incomplete, clearly say that the available information is limited.
7. If symptoms sound serious, recommend consulting a healthcare professional.
8. Do not prescribe medicines or dosages.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text