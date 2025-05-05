from transformers import pipeline

# Initialize the sentiment analysis pipeline with a Spanish model
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="UMUTeam/roberta-spanish-sentiment-analysis"
)


def analyze_sentiment(text_input):
    """
    Analyzes the sentiment of the input text (string or list of strings)
    and prints the result.
    """
    print(f"--- Analyzing Sentiment ---")
    print(f"Input: {text_input}")
    result = sentiment_analyzer(text_input)
    print(f"Result: {result}")
    print("-" * 30)


# Run the analysis function with different examples
if __name__ == "__main__":
    # Positive example
    analyze_sentiment(
        "¡Qué día tan maravilloso! Estoy muy contento con los resultados."
    )

    # Negative example
    analyze_sentiment("Este servicio es terrible, estoy muy decepcionado.")

    # Neutral-like example
    analyze_sentiment("El informe se entregará mañana por la tarde.")

    # Batch example
    analyze_sentiment(
        [
            "Me encanta este lugar, la comida es deliciosa.",
            "El tráfico esta mañana fue horrible.",
            "La película estuvo bien, ni buena ni mala.",
            "¡Felicidades por tu nuevo trabajo!",
        ]
    )

# To run this script:
# 1. Make sure you have transformers and torch (or tensorflow) installed:
#    pip install transformers torch
#    or
#    pip install transformers tensorflow
# 2. Run the script from your terminal:
#    python test_transformers.py
