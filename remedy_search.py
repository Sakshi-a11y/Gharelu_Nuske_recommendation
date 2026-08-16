import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import re


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("clean_remedies.csv")

print("Dataset loaded:", df.shape)


# ==========================================
# 2. CLEAN TEXT
# ==========================================

# Convert columns to string
df["Health Issue"] = df["Health Issue"].fillna("").astype(str)
df["Home Remedy"] = df["Home Remedy"].fillna("").astype(str)
df["Yogasan"] = df["Yogasan"].fillna("").astype(str)


# Remove unwanted URLs from Yogasan
df["Yogasan"] = (
    df["Yogasan"]
    .str.replace(r"https?://\S+", "", regex=True)
    .str.strip()
)


# ==========================================
# 3. REMOVE CONTAMINATED TEXT
# ==========================================

df["Home Remedy"] = (
    df["Home Remedy"]
    .str.replace(
        r"Dog-bite.*",
        "",
        regex=True,
        flags=re.IGNORECASE
    )
    .str.strip()
)


# ==========================================
# 4. CREATE TEXT FOR EMBEDDING
# ==========================================

df["search_text"] = (
    "Health Issue: " + df["Health Issue"] +
    ". Home Remedy: " + df["Home Remedy"]
)


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ==========================================
# 6. CREATE EMBEDDINGS
# ==========================================

print("Creating embeddings...")

embeddings = model.encode(
    df["search_text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# Normalize embeddings
faiss.normalize_L2(embeddings)


# ==========================================
# 7. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index created.")
print("Total records:", index.ntotal)


# ==========================================
# 8. SEARCH FUNCTION
# ==========================================

def search_remedy(query, top_k=8):

    # --------------------------------------
    # Convert user query into embedding
    # --------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Normalize query
    faiss.normalize_L2(query_embedding)


    # --------------------------------------
    # Search FAISS
    # --------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )


    # --------------------------------------
    # Store grouped results
    # --------------------------------------

    grouped_results = {}


    # --------------------------------------
    # Process retrieved records
    # --------------------------------------

    for score, idx in zip(scores[0], indices[0]):

        score = float(score)

        # Ignore weak matches
        if score < 0.45:
            continue

        row = df.iloc[idx]

        health_issue = row["Health Issue"]


        # Create health issue entry
        if health_issue not in grouped_results:

            grouped_results[health_issue] = {
                "score": score,
                "remedies": [],
                "yogasan": []
            }


        # ----------------------------------
        # Add Home Remedy
        # ----------------------------------

        remedy = row["Home Remedy"]

        if (
            remedy
            and remedy not in grouped_results[health_issue]["remedies"]
        ):
            grouped_results[health_issue]["remedies"].append(remedy)


        # ----------------------------------
        # Add Yogasan
        # ----------------------------------

        yoga = row["Yogasan"]

        if (
            yoga
            and yoga not in grouped_results[health_issue]["yogasan"]
        ):
            grouped_results[health_issue]["yogasan"].append(yoga)


    # ======================================
    # RETURN RESULTS
    # ======================================

    return grouped_results


# ==========================================
# 9. FORMAT RESULTS FOR RAG
# ==========================================

def get_remedy_context(query, top_k=8):

    results = search_remedy(query, top_k)

    if not results:
        return ""


    context = ""


    for issue, data in results.items():

        context += f"Health Issue: {issue}\n"

        context += (
            f"Similarity Score: "
            f"{round(data['score'], 3)}\n"
        )


        # Add remedies
        for remedy in data["remedies"][:2]:

            context += f"Home Remedy: {remedy}\n"


        # Add yoga
        for yoga in data["yogasan"][:1]:

            context += f"Yogasan: {yoga}\n"


        context += "\n"


    return context


# ==========================================
# 10. TESTING
# ==========================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nEnter your health problem "
            "(type 'exit' to stop): "
        )


        if query.lower().strip() == "exit":
            print("Exiting...")
            break


        # Search FAISS
        results = search_remedy(query)


        # ----------------------------------
        # Display Results
        # ----------------------------------

        print("\n====================================")
        print("USER QUERY")
        print("====================================")

        print(query)


        if not results:

            print("\nNo relevant remedy found.")

            continue


        print("\n====================================")
        print("RELEVANT HEALTH ISSUES")
        print("====================================")


        for issue, data in results.items():

            print("\nHealth Issue:", issue)

            print(
                "Best Similarity:",
                round(data["score"], 3)
            )


            print("\nHome Remedies:")

            for remedy in data["remedies"][:2]:

                print("-", remedy)


            if data["yogasan"]:

                print("\nYogasan:")

                for yoga in data["yogasan"][:1]:

                    print("-", yoga)


        # ----------------------------------
        # Show RAG Context
        # ----------------------------------

        context = get_remedy_context(query)


        print("\n====================================")
        print("RAG CONTEXT")
        print("====================================")

        print(context)