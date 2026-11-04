import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_FOLDER = "data"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

print("\nLoading datasets...")

faqs = pd.read_csv(os.path.join(DATA_FOLDER, "faqs.csv"))
products = pd.read_csv(os.path.join(DATA_FOLDER, "products.csv"))

print(f"FAQ records     : {len(faqs)}")
print(f"Product records : {len(products)}")

# --------------------------------------------------
# FAQ TEXT
# --------------------------------------------------

faq_texts = (
    faqs["Question"].fillna("").astype(str)
    + " "
    + faqs["Answer"].fillna("").astype(str)
).tolist()

# --------------------------------------------------
# PRODUCT TEXT
# --------------------------------------------------

product_texts = (
    products["ProductName"].fillna("").astype(str)
    + " "
    + products["Category"].fillna("").astype(str)
    + " "
    + products["Brand"].fillna("").astype(str)
    + " "
    + products["Description"].fillna("").astype(str)
).tolist()

# --------------------------------------------------
# CREATE FAQ EMBEDDINGS
# --------------------------------------------------

print("\nCreating FAQ embeddings...")

faq_embeddings = model.encode(
    faq_texts,
    show_progress_bar=True
)

print("FAQ embeddings created!")
print("FAQ embedding shape:", faq_embeddings.shape)

# --------------------------------------------------
# CREATE PRODUCT EMBEDDINGS
# --------------------------------------------------

print("\nCreating product embeddings...")

product_embeddings = model.encode(
    product_texts,
    show_progress_bar=True
)

print("Product embeddings created!")
print("Product embedding shape:", product_embeddings.shape)

# --------------------------------------------------
# SAVE FAQ EMBEDDINGS
# --------------------------------------------------

faq_data = {
    "texts": faq_texts,
    "embeddings": faq_embeddings,
    "data": faqs
}

with open(
    os.path.join(MODEL_FOLDER, "faq_embeddings.pkl"),
    "wb"
) as file:
    pickle.dump(faq_data, file)

# --------------------------------------------------
# SAVE PRODUCT EMBEDDINGS
# --------------------------------------------------

product_data = {
    "texts": product_texts,
    "embeddings": product_embeddings,
    "data": products
}

with open(
    os.path.join(MODEL_FOLDER, "product_embeddings.pkl"),
    "wb"
) as file:
    pickle.dump(product_data, file)

# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

print("\n" + "=" * 70)
print("EMBEDDING CREATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCreated files:")

print("models/faq_embeddings.pkl")
print("models/product_embeddings.pkl")
