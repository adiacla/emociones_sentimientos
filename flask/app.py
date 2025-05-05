from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from sentiment_analysis_spanish import sentiment_analysis  # Usamos sentiment_analysis_spanish

# Import functions and variables from our new modules
import config
import utils
import model_loader

app = Flask(__name__)

# --- Load Artifacts at Startup ---
tokenizer = model_loader.load_tokenizer()
label_encoder, EMOTION_LABELS = model_loader.load_label_encoder()
model = model_loader.load_keras_model()

# --- Initialize Sentiment Analyzer ---
sentiment_analyzer = sentiment_analysis.SentimentAnalysisSpanish()

# --- Routes ---

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_data_url = data.get("image")
    input_text = data.get("text")

    ocr_text = None

    # --- Input Handling ---
    if image_data_url:
        if not image_data_url.startswith("data:image/png;base64,"):
            return jsonify({"error": "Invalid image data URL format"}), 400

        try:
            base64_string = image_data_url.split(",")[1]
            image_bytes = base64.b64decode(base64_string)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return jsonify({"error": "Failed to decode image"}), 500

        ocr_text = utils.perform_ocr(image_bytes)

        if ocr_text is None:
            return jsonify({
                "text": "(Error OCR)",
                "prediction": "Fallo el reconocimiento de texto (OCR)",
            }), 200
        if not ocr_text:
            return jsonify({
                "text": "",
                "prediction": "No se detectó texto en la imagen"
            }), 200

    elif input_text is not None:
        ocr_text = input_text
        if not ocr_text:
            return jsonify({"text": "", "prediction": "No se proporcionó texto"}), 200
    else:
        return jsonify({"error": "Request must contain either 'image' or 'text'"}), 400

    # --- Sentiment Analysis before Emotion Prediction ---
    sentiment_score = sentiment_analyzer.sentiment(ocr_text)

    # Define the sentiment label based on the score
    if sentiment_score > 0.6:
        sentiment_label = "positiva"  # Positive
    elif sentiment_score < 0.5:
        sentiment_label = "negativa"  # Negative
    else:
        sentiment_label = "neutral"  # Neutral

    if sentiment_label in ["positiva", "neutral"]:
        return jsonify({
            "text": ocr_text,
            "prediction": f"El texto tiene un sentimiento con tendencia {sentiment_label.lower()} y no se realiza el análisis emocional."
        }), 200

    # --- Emotion Prediction ---
    prediction_result = "Model/Tokenizer/Encoder not loaded or preprocessing failed"

    if model and tokenizer and label_encoder and EMOTION_LABELS:
        processed_text = utils.preprocess_text_for_model(ocr_text, tokenizer)

        if processed_text is not None and processed_text.size > 0:
            try:
                predictions = model.predict(processed_text)[0]
                probs = {
                    label: float(p) for label, p in zip(EMOTION_LABELS, predictions)
                }
                idx = np.argmax(predictions)
                pred_emotion = label_encoder.inverse_transform([idx])[0]

                prediction_result = {
                    "emotion": pred_emotion,
                    "confidence": float(predictions[idx]),
                    "all_probabilities": probs,
                }
                print(f"Input Text: '{ocr_text}' -> Prediction: {prediction_result}")

            except Exception as e:
                print(f"Error during model prediction: {e}")
                prediction_result = f"Predicción fallida: {type(e).__name__}"
        else:
            prediction_result = "El texto quedó vacío después de la limpieza o el preprocesamiento falló"
            print(f"Text preprocessing returned None or empty array for input: '{ocr_text}'")
    else:
        missing = []
        if not model:
            missing.append("Modelo")
        if not tokenizer:
            missing.append("Tokenizer")
        if not label_encoder:
            missing.append("Codificador de Etiquetas")
        if not EMOTION_LABELS:
            missing.append("Etiquetas de Emoción")
        prediction_result = f"No se puede predecir, faltan componentes: {', '.join(missing)}"

    return jsonify({"text": ocr_text, "prediction": prediction_result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
