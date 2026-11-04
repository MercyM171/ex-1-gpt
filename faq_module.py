import os
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FAQ MODULE
# Semantic Search + Cosine Similarity + Similarity Threshold
# + Ollama Natural Language Response
# ============================================================

print("Loading FAQ Module...")

MODEL_FOLDER = "models"

FAQ_FILE = os.path.join(
    MODEL_FOLDER,
    "faq_embeddings.pkl"
)

# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# Load FAQ embeddings
# ------------------------------------------------------------

with open(FAQ_FILE, "rb") as file:
    faq_data = pickle.load(file)

faq_embeddings = faq_data["embeddings"]
faq_dataframe = faq_data["data"]


# ------------------------------------------------------------
# FAQ SEARCH FUNCTION
# ------------------------------------------------------------

def get_faq_answer(
    user_question,
    threshold=0.35
):

    # Convert user question into embedding
    query_embedding = embedding_model.encode(
        [user_question]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    # Find best matching FAQ
    best_index = np.argmax(similarities)

    best_score = similarities[best_index]

    best_faq = faq_dataframe.iloc[
        best_index
    ]

    # --------------------------------------------------------
    # Similarity Threshold
    # --------------------------------------------------------

    if best_score < threshold:

        return {
            "matched": False,
            "answer": None,
            "question": None,
            "score": float(best_score)
        }

    # --------------------------------------------------------
    # Matched FAQ
    # --------------------------------------------------------

    faq_question = str(
        best_faq["Question"]
    )

    faq_answer = str(
        best_faq["Answer"]
    )

    # --------------------------------------------------------
    # Ollama Natural Language Response
    # --------------------------------------------------------

    prompt = f"""
You are an e-commerce customer support chatbot.

Answer the user's question naturally and helpfully.

Use ONLY the information provided in the FAQ answer.
Do not invent information.

User Question:
{user_question}

FAQ Question:
{faq_question}

FAQ Answer:
{faq_answer}

Give a short, friendly and natural response.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    natural_answer = response["message"]["content"]

    return {
        "matched": True,
        "answer": natural_answer,
        "question": faq_question,
        "score": float(best_score)
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FAQ SEMANTIC SEARCH")
    print("=" * 70)

    while True:

        question = input(
            "\nAsk Question : "
        ).strip()

        if question.lower() == "exit":

            print("\nExiting FAQ Module...")
            break

        if not question:

            print("Please enter a question.")
            continue

        result = get_faq_answer(
            question
        )

        # ----------------------------------------------------
        # MATCH FOUND
        # ----------------------------------------------------

        if result["matched"]:

            print(
                "\nBest Matching FAQ : "
                + result["question"]
            )

            print(
                f"Similarity Score : "
                f"{result['score']:.2f}"
            )

            print(
                "\nChatbot: "
                + result["answer"]
            )

        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        else:

            print(
                "\nNo relevant FAQ found."
            )

            print(
                f"Similarity Score : "
                f"{result['score']:.2f}"
            )

            print(
                "\nContinuing to the next module..."
            )
