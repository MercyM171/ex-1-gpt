import pandas as pd
import os

DATA_FILE = os.path.join(
    "data",
    "products.csv"
)

products = pd.read_csv(DATA_FILE)


# --------------------------------------------------
# PRODUCT RECOMMENDATION
# --------------------------------------------------

def recommend_products(
    query="",
    max_price=None,
    category=None,
    brand=None,
    min_rating=None,
    top_k=5
):

    filtered = products.copy()

    # --------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------

    if max_price is not None:

        filtered = filtered[
            filtered["Price"] <= max_price
        ]

    # --------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------

    if category:

        filtered = filtered[
            filtered["Category"]
            .str.contains(
                category,
                case=False,
                na=False
            )
        ]

    # --------------------------------------------------
    # BRAND FILTER
    # --------------------------------------------------

    if brand:

        filtered = filtered[
            filtered["Brand"]
            .str.contains(
                brand,
                case=False,
                na=False
            )
        ]

    # --------------------------------------------------
    # RATING FILTER
    # --------------------------------------------------

    if min_rating is not None:

        filtered = filtered[
            filtered["Rating"] >= min_rating
        ]

    # --------------------------------------------------
    # SORT BY RATING
    # --------------------------------------------------

    filtered = filtered.sort_values(
        by="Rating",
        ascending=False
    )

    filtered = filtered.head(top_k)

    # --------------------------------------------------
    # CREATE RECOMMENDATIONS
    # --------------------------------------------------

    recommendations = []

    for _, product in filtered.iterrows():

        reason = (
            f"{product['ProductName']} is recommended "
            f"because it has a rating of "
            f"{product['Rating']} and costs "
            f"₹{product['Price']}."
        )

        recommendations.append({
            "Product": product["ProductName"],
            "Brand": product["Brand"],
            "Category": product["Category"],
            "Price": product["Price"],
            "Rating": product["Rating"],
            "Reason": reason
        })

    return recommendations


# --------------------------------------------------
# FORMAT RECOMMENDATIONS
# --------------------------------------------------

def format_recommendations(
    recommendations
):

    if not recommendations:

        return (
            "Sorry, I could not find products "
            "matching your requirements."
        )

    response = "Here are my recommendations:\n\n"

    for i, product in enumerate(
        recommendations,
        start=1
    ):

        response += (
            f"{i}. {product['Product']}\n"
            f"   Brand: {product['Brand']}\n"
            f"   Category: {product['Category']}\n"
            f"   Price: ₹{product['Price']}\n"
            f"   Rating: {product['Rating']}\n"
            f"   Reason: {product['Reason']}\n\n"
        )

    return response
