import pandas as pd
import os

REVIEW_FILE = os.path.join(
    "data",
    "reviews.csv"
)

reviews = pd.read_csv(REVIEW_FILE)


# --------------------------------------------------
# REVIEW SUMMARY
# --------------------------------------------------

def summarize_reviews(
    product_name=None,
    product_id=None
):

    data = reviews.copy()

    # --------------------------------------------------
    # FILTER BY PRODUCT ID
    # --------------------------------------------------

    if product_id is not None:

        data = data[
            data["ProductID"] == product_id
        ]

    # --------------------------------------------------
    # FILTER BY PRODUCT NAME
    # --------------------------------------------------

    elif product_name:

        data = data[
            data["ProductName"]
            .str.contains(
                product_name,
                case=False,
                na=False
            )
        ]

    if data.empty:

        return (
            "Sorry, no reviews were found "
            "for this product."
        )

    # --------------------------------------------------
    # RATING COUNTS
    # --------------------------------------------------

    average_rating = data["Rating"].mean()

    positive = len(
        data[data["Rating"] >= 4]
    )

    neutral = len(
        data[data["Rating"] == 3]
    )

    negative = len(
        data[data["Rating"] <= 2]
    )

    total = len(data)

    # --------------------------------------------------
    # SENTIMENT PERCENTAGES
    # --------------------------------------------------

    positive_percentage = (
        positive / total
    ) * 100

    negative_percentage = (
        negative / total
    ) * 100

    # --------------------------------------------------
    # SAMPLE REVIEWS
    # --------------------------------------------------

    positive_reviews = data[
        data["Rating"] >= 4
    ]["Review"].head(3).tolist()

    negative_reviews = data[
        data["Rating"] <= 2
    ]["Review"].head(3).tolist()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    summary = (
        f"Review Summary\n"
        f"-------------------------\n"
        f"Product: {data.iloc[0]['ProductName']}\n"
        f"Total Reviews: {total}\n"
        f"Average Rating: {average_rating:.2f}/5\n"
        f"Positive Reviews: {positive} "
        f"({positive_percentage:.1f}%)\n"
        f"Neutral Reviews: {neutral}\n"
        f"Negative Reviews: {negative} "
        f"({negative_percentage:.1f}%)\n"
    )

    if positive_reviews:

        summary += "\nExample Positive Reviews:\n"

        for review in positive_reviews:

            summary += f"- {review}\n"

    if negative_reviews:

        summary += "\nExample Negative Reviews:\n"

        for review in negative_reviews:

            summary += f"- {review}\n"

    return summary
