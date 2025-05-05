import os
import pickle
from tensorflow import keras
import config  # Import paths from config


def load_tokenizer():
    """Loads the tokenizer from the path specified in config."""
    print(f"--- Loading Tokenizer ---")
    print(f"Attempting to load from: {config.TOKENIZER_PATH}")
    tokenizer = None
    try:
        if not os.path.exists(config.TOKENIZER_PATH):
            print(f"ERROR: Tokenizer file NOT FOUND at {config.TOKENIZER_PATH}")
            return None
        if not os.access(config.TOKENIZER_PATH, os.R_OK):
            print(
                f"ERROR: No read permission for Tokenizer file at {config.TOKENIZER_PATH}"
            )
            return None

        with open(config.TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded successfully.")
    except (pickle.UnpicklingError, EOFError) as pe:
        print(f"ERROR: Failed to unpickle Tokenizer. File might be corrupted.")
        print(f"Specific error: {type(pe).__name__}: {pe}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading tokenizer.")
        print(f"Exception type: {type(e).__name__}: {e}")
    finally:
        print(f"--- Finished Loading Tokenizer ---")
        return tokenizer


def load_label_encoder():
    """Loads the label encoder from the path specified in config with enhanced debugging."""
    print(f"--- Loading Label Encoder ---")
    print(f"Attempting to load from: {config.LABEL_ENCODER_PATH}")
    label_encoder = None
    emotion_labels = []

    # 1. Check if file exists
    file_exists = os.path.exists(config.LABEL_ENCODER_PATH)
    print(f"Does file exist at path? {file_exists}")

    if file_exists:
        # 2. Check for read permissions
        has_read_permission = os.access(config.LABEL_ENCODER_PATH, os.R_OK)
        print(f"Does the process have read permission? {has_read_permission}")

        if has_read_permission:
            try:
                with open(config.LABEL_ENCODER_PATH, "rb") as handle:
                    print(f"Attempting to open and pickle.load()...")
                    label_encoder = pickle.load(handle)
                print(f"Label Encoder loaded successfully.")
                # Get emotion labels directly from the encoder
                emotion_labels = label_encoder.classes_.tolist()
                print(f"Loaded emotion labels: {emotion_labels}")
            except FileNotFoundError:
                print(
                    f"ERROR: FileNotFoundError encountered unexpectedly."
                )  # Should not happen after exists check
            except (pickle.UnpicklingError, EOFError) as pe:
                print(
                    f"ERROR: Failed to unpickle Label Encoder. File might be corrupted or incomplete."
                )
                print(f"Specific unpickling error: {type(pe).__name__}: {pe}")
                label_encoder = None  # Ensure it's None on error
            except Exception as e:
                print(f"ERROR: An unexpected error occurred loading label encoder.")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception details: {e}")
                label_encoder = None  # Ensure it's None on error
        else:
            print(f"ERROR: No read permission for Label Encoder file.")
    else:
        print(f"ERROR: Label Encoder file confirmed NOT FOUND.")

    print(f"--- Finished Loading Label Encoder ---")
    return label_encoder, emotion_labels  # Return both


def load_keras_model():
    """Loads the Keras model, preferring Phase 2."""
    print(f"--- Loading Keras Model ---")
    model = None
    model_path_to_load = None

    # Determine which model file to load (prefer Phase 2)
    if os.path.exists(config.MODEL_PATH_FASE2):
        model_path_to_load = config.MODEL_PATH_FASE2
        print(f"Using Phase 2 model: {model_path_to_load}")
    elif os.path.exists(config.MODEL_PATH_FASE1):
        model_path_to_load = config.MODEL_PATH_FASE1
        print(f"Using Phase 1 model (fallback): {model_path_to_load}")
    else:
        print(
            f"ERROR: Neither model file found at {config.MODEL_PATH_FASE2} nor {config.MODEL_PATH_FASE1}."
        )
        print(f"--- Finished Loading Keras Model ---")
        return None  # No model path found

    # Load the Keras model
    try:
        if not os.access(model_path_to_load, os.R_OK):
            print(f"ERROR: No read permission for Model file at {model_path_to_load}")
            print(f"--- Finished Loading Keras Model ---")
            return None

        model = keras.models.load_model(model_path_to_load)
        print(f"Model loaded successfully from {model_path_to_load}")
        # model.summary() # Optional
    except FileNotFoundError:
        # Should not happen after exists check, but good practice
        print(
            f"ERROR: Model file not found at {model_path_to_load} (unexpected, check permissions or race condition)"
        )
    except Exception as e:
        print(f"Error loading Keras model from {model_path_to_load}: {e}")
        print(f"Exception type: {type(e).__name__}")

    print(f"--- Finished Loading Keras Model ---")
    return model
