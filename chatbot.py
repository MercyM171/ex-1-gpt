from intent_module import detect_intent
from faq_module import get_faq_answer
from product_module import answer_product_question
from recommendation_module import (
    recommend_products,
    format_recommendations
)
from review_module import summarize_reviews


# --------------------------------------------------
# DISPLAY HEADER
# --------------------------------------------------

def display_header():

    print("\n" + "=" * 70)
    print("             ECOMMERCEGPT CHATBOT")
    print("=" * 70)

    print(
        "\nI can help you with:"
    )

    print("1. Frequently Asked Questions")
    print("2. Product Information")
    print("3. Product Recommendations")
    print("4. Customer Reviews")

    print(
        "\nType 'exit' to close the chatbot."
    )

    print("=" * 70)


# --------------------------------------------------
# FAQ HANDLER
# --------------------------------------------------

def handle_faq(user_input):

    result = get_faq_answer(
        user_input
    )

    print("\nBot:")
    print(result["answer"])

    if result["question"]:

        print(
            f"\nMatched FAQ: "
            f"{result['question']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.3f}"
        )


# --------------------------------------------------
# PRODUCT HANDLER
# --------------------------------------------------

def handle_product(user_input):

    answer = answer_product_question(
        user_input
    )

    print("\nBot:")
    print(answer)


# --------------------------------------------------
# RECOMMENDATION HANDLER
# --------------------------------------------------

def handle_recommendation(user_input):

    recommendations = recommend_products(
        query=user_input,
        top_k=5
    )

    response = format_recommendations(
        recommendations
    )

    print("\nBot:")
    print(response)


# --------------------------------------------------
# REVIEW HANDLER
# --------------------------------------------------

def handle_review(user_input):

    summary = summarize_reviews(
        product_name=user_input
    )

    print("\nBot:")
    print(summary)


# --------------------------------------------------
# MAIN CHATBOT
# --------------------------------------------------

def chatbot():

    display_header()

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        if not user_input:

            print(
                "Bot: Please enter a question."
            )

            continue

        # --------------------------------------------------
        # INTENT DETECTION
        # --------------------------------------------------

        intent = detect_intent(
            user_input
        )

        print(
            f"\n[Detected Intent: {intent}]"
        )

        # --------------------------------------------------
        # ROUTING
        # --------------------------------------------------

        if intent == "exit":

            print(
                "\nBot: Thank you for using "
                "ECommerceGPT! Goodbye! ❤️"
            )

            break

        elif intent == "faq":

            handle_faq(
                user_input
            )

        elif intent == "recommendation":

            handle_recommendation(
                user_input
            )

        elif intent == "review":

            handle_review(
                user_input
            )

        elif intent == "product":

            handle_product(
                user_input
            )


# --------------------------------------------------
# PROGRAM START
# --------------------------------------------------

if __name__ == "__main__":

    chatbot()
