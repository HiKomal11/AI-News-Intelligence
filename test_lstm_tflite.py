from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.preprocessing.text_cleaner import clean_text


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "lstm_news_classifier.tflite"
)

TOKENIZER_FILE = (
    BASE_DIR
    / "models"
    / "lstm_tokenizer.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_SEQUENCE_LENGTH = 200


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = joblib.load(
    TOKENIZER_FILE
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_FILE)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("TFLite model loaded successfully.")

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


# ============================================================
# TEST NEWS
# ============================================================

text = """
The government announced a new national program today
to improve digital education and provide students with
better access to online learning resources across the country.
"""


print("\nOriginal text:")
print(text)


# ============================================================
# CLEAN TEXT
# ============================================================

cleaned_text = clean_text(
    text
)


print("\nCleaned text:")
print(cleaned_text)


# ============================================================
# TOKENIZE
# ============================================================

sequence = tokenizer.texts_to_sequences(
    [cleaned_text]
)[0]


# ============================================================
# PAD
# ============================================================

padded = np.zeros(
    (
        1,
        MAX_SEQUENCE_LENGTH
    ),
    dtype=np.int32
)


sequence = sequence[
    :MAX_SEQUENCE_LENGTH
]


if sequence:

    padded[
        0,
        :len(sequence)
    ] = sequence


# ============================================================
# RUN TFLITE
# ============================================================

print("\nRunning TFLite prediction...")

interpreter.set_tensor(
    input_details[0]["index"],
    padded
)

interpreter.invoke()


output = interpreter.get_tensor(
    output_details[0]["index"]
)


probability = float(
    output[0][0]
)


# ============================================================
# RESULT
# ============================================================

if probability >= 0.5:

    label = "REAL"

    confidence = probability * 100

else:

    label = "FAKE"

    confidence = (1 - probability) * 100


print("\n======================================")
print("        TFLITE LSTM RESULT")
print("======================================")

print(
    "Prediction:",
    label
)

print(
    "Probability:",
    round(probability, 4)
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("======================================")