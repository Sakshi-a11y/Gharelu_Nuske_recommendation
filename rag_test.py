from remedy_search import get_remedy_context
from gemini_service import generate_remedy_response


query = input("Enter your health problem: ")


# ==========================================
# RETRIEVAL
# ==========================================

context = get_remedy_context(query)


if not context:

    print("\nNo relevant information found in knowledge base.")

else:

    # ======================================
    # GENERATION
    # ======================================

    response = generate_remedy_response(
        query,
        context
    )

    print("\n====================================")
    print("AI RESPONSE")
    print("====================================")

    print(response)