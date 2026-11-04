import pandas as pd

# ============================================
# EXPERIMENT 1 - Q1
# Dataset Preparation
# ============================================

# Load the datasets
faqs = pd.read_csv("data/faqs.csv")
products = pd.read_csv("data/products.csv")
reviews = pd.read_csv("data/reviews.csv")


# Function to display dataset information
def display_dataset(name, data):

    print("\n" + "=" * 70)
    print(f"{name} DATASET")
    print("=" * 70)

    # First five records
    print("\nFirst 5 Records:")
    print(data.head())

    # Number of rows and columns
    print("\nNumber of Rows and Columns:")
    print(data.shape)

    # Column names / attributes
    print("\nAttributes / Columns:")
    print(data.columns.tolist())


# Display information about all three datasets
display_dataset("FAQ", faqs)
display_dataset("PRODUCT", products)
display_dataset("REVIEW", reviews)
