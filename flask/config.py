import os

# Get the directory where the script is located
basedir = os.path.abspath(os.path.dirname(__file__))

# --- Artifact Paths ---
TOKENIZER_PATH = os.path.join(basedir, "tokenizer_emociones.pickle")
LABEL_ENCODER_PATH = os.path.join(basedir, "label_encoder_emociones.pickle")
MODEL_PATH_FASE2 = os.path.join(basedir, "best_model_emociones_emb_fase2.keras")
MODEL_PATH_FASE1 = os.path.join(
    basedir, "best_model_emociones_emb_fase1.keras"
)  # Fallback

# --- Preprocessing Parameters ---
MAX_SEQUENCE_LENGTH = 50  # From your notebook
