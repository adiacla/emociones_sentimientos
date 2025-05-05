import re
import unicodedata
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_SEQUENCE_LENGTH
from rapidocr import RapidOCR
from PIL import Image
import io
import tempfile
import os
import traceback

# Initialize RapidOCR engine
engine = RapidOCR()

# Monkey‑patch DynamicCache so chat() won’t blow up on get_max_length
try:
    from accelerate.utils import DynamicCache

    if not hasattr(DynamicCache, "get_max_length"):

        def get_max_length(self):
            # accelerate’s cache may expose cache_size or max_size
            return getattr(self, "cache_size", None) or getattr(self, "max_size", None)

        DynamicCache.get_max_length = get_max_length
except ImportError:
    pass


# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Perform OCR using RapidOCR
def perform_ocr(img_bytes):
    """Performs OCR on the image bytes using RapidOCR."""
    if not img_bytes:
        print("OCR skipped: Input image bytes are empty.")
        return None

    temp_file_path = None
    try:
        # Load image and save to temp file for RapidOCR
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            temp_file_path = tmp.name

        # Run RapidOCR - it returns a RapidOCROutput object
        result_object = engine(temp_file_path)

        # Check if the result object exists and has the 'txts' attribute
        if result_object and hasattr(result_object, "txts"):
            ocr_lines = result_object.txts  # Access the 'txts' attribute

            # Check if the txts attribute is not None and contains text
            if ocr_lines:
                result_text = " ".join(ocr_lines)  # Join the list/tuple of strings
                print(f"RapidOCR Result: '{result_text}'")
                return result_text.strip()
            else:
                print("RapidOCR returned an empty 'txts' attribute.")
                return ""
        else:
            # Handle cases where engine() returned None or an object without 'txts'
            print("RapidOCR engine returned no result object or object lacks 'txts'.")
            print(f"Result object was: {result_object}")
            return ""  # Or None, depending on desired behavior

    except Exception as e:
        print(f"Error during OCR with RapidOCR: {e}")
        traceback.print_exc()
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as oe:
                print(f"Error removing temporary file {temp_file_path}: {oe}")


# Preprocess text for the model
def preprocess_text_for_model(text, tokenizer):
    """Preprocesses text using loaded tokenizer and padding."""
    if not text or not tokenizer:
        print("Preprocessing failed: No text or tokenizer.")
        return None
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            print("Preprocessing failed: Text empty after cleaning.")
            return None

        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",  # Use imported constant
        )

        print(
            f"Original OCR: '{text}' -> Cleaned: '{cleaned_text}' -> Padded Shape: {padded_sequences.shape}"
        )
        return padded_sequences

    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return None
