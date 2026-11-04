# --------------------------------------------------
# INTENT CLASSIFICATION MODULE
# --------------------------------------------------

def detect_intent(user_input):

    text = user_input.lower().strip()

    # --------------------------------------------------
    # EXIT
    # --------------------------------------------------

    if text in [
        "exit",
        "quit",
        "bye",
        "goodbye"
    ]:

        return "exit"

    # --------------------------------------------------
    # FAQ INTENT
    # --------------------------------------------------

    faq_keywords = [
        "how do i",
        "how can i",
        "where is my",
        "track my",
        "return",
        "cancel",
        "delivery",
        "refund",
        "payment",
        "order"
    ]

    if any(
        keyword in text
        for keyword in faq_keywords
    ):

        return "faq"

    # --------------------------------------------------
    # RECOMMENDATION INTENT
    # --------------------------------------------------

    recommendation_keywords = [
        "recommend",
        "suggest",
        "best",
        "which one",
        "what should i buy",
        "good product",
        "under",
        "budget"
    ]

    if any(
        keyword in text
        for keyword in recommendation_keywords
    ):

        return "recommendation"

    # --------------------------------------------------
    # REVIEW INTENT
    # --------------------------------------------------

    review_keywords = [
        "review",
        "reviews",
        "customers think",
        "feedback",
        "opinion",
        "rating",
        "ratings"
    ]

    if any(
        keyword in text
        for keyword in review_keywords
    ):

        return "review"

    # --------------------------------------------------
    # PRODUCT INTENT
    # --------------------------------------------------

    product_keywords = [
        "product",
        "price",
        "cost",
        "features",
        "specification",
        "specifications",
        "description",
        "brand",
        "laptop",
        "phone",
        "mobile",
        "headphone",
        "watch",
        "camera"
    ]

    if any(
        keyword in text
        for keyword in product_keywords
    ):

        return "product"

    # --------------------------------------------------
    # DEFAULT
    # --------------------------------------------------

    return "product"
