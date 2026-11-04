import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FOLDER = "models"

PRODUCT_FILE = os.path.join(
    MODEL_FOLDER,
    "product_embeddings.pkl"
)

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------------------------
# LOAD PRODUCT DATA
# --------------------------------------------------

with open(PRODUCT_FILE, "rb") as file:
    product_data = pickle.load(file)

product_embeddings = product_data["embeddings"]
products = product_data["data"]


# --------------------------------------------------
# PRODUCT RETRIEVAL
# --------------------------------------------------

def retrieve_products(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        product_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    results = []

    for index in top_indices:

        product = products.iloc[index]

        results.append({
            "ProductID": product["ProductID"],
            "ProductName": product["ProductName"],
            "Category": product["Category"],
            "Brand": product["Brand"],
            "Price": product["Price"],
            "Description": product["Description"],
            "Rating": product["Rating"],
            "Similarity": float(
                similarities[index]
            )
        })

    return results


# --------------------------------------------------
# RAG PRODUCT ANSWER
# --------------------------------------------------

def answer_product_question(query):

    results = retrieve_products(
        query,
        top_k=3
    )

    if not results:
        return "Sorry, no relevant products were found."

    answer = "Based on the available product information:\n\n"

    for product in results:

        answer += (
            f"Product: {product['ProductName']}\n"
            f"Brand: {product['Brand']}\n"
            f"Category: {product['Category']}\n"
            f"Price: ₹{product['Price']}\n"
            f"Rating: {product['Rating']}\n"
            f"Description: {product['Description']}\n"
            f"Relevance Score: "
            f"{product['Similarity']:.3f}\n\n"
        )

    return answer
