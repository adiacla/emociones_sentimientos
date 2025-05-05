from sentiment_analysis_spanish import sentiment_analysis

def analizar_sentimiento(texto):
    analizador = sentiment_analysis.SentimentAnalysisSpanish()
    puntuacion = analizador.sentiment(texto)

    # Clasificación simple basada en el puntaje
    if puntuacion >= 0.6:
        sentimiento = "POSITIVO"
    elif puntuacion <= 0.4:
        sentimiento = "NEGATIVO"
    else:
        sentimiento = "NEUTRO"

    return sentimiento, puntuacion

# Ejemplo de uso
frase = "Me encanta este producto, es excelente."
resultado, score = analizar_sentimiento(frase)
print(f"Texto: {frase}")
print(f"Sentimiento: {resultado} (puntuación: {score:.3f})")
