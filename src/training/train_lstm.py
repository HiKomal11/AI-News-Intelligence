import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = BASE_DIR / "data" / "processed_news_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MAX_WORDS = 30000
MAX_SEQUENCE_LENGTH = 300

EMBEDDING_DIM = 128
LSTM_UNITS = 64

TEST_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 8
BATCH_SIZE = 64


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed dataset...")

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["clean_text", "label"]
)

print("Dataset shape:", df.shape)


X = df["clean_text"].astype(str).values
y = df["label"].astype(int).values


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# TOKENIZATION
# ============================================================

print("\nCreating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


print("Vocabulary size:", len(tokenizer.word_index))


# ============================================================
# TEXT → SEQUENCES
# ============================================================

print("\nConverting text to sequences...")

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)


# ============================================================
# PADDING
# ============================================================

print("Padding sequences...")

X_train_pad = pad_sequences(
    X_train_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


print("Training tensor shape:", X_train_pad.shape)
print("Testing tensor shape:", X_test_pad.shape)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([
    
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),

    LSTM(
        LSTM_UNITS
    ),

    Dropout(
        0.5
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.3
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\n========== MODEL SUMMARY ==========")

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n========== TRAINING LSTM ==========")

history = model.fit(
    X_train_pad,
    y_train,

    validation_split=0.10,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

probabilities = model.predict(
    X_test_pad,
    batch_size=BATCH_SIZE
)

y_pred = (
    probabilities >= 0.5
).astype(int).flatten()


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== LSTM RESULTS ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Fake",
        "Real"
    ]
)

disp.plot()

plt.title(
    "LSTM Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    BASE_DIR / "lstm_confusion_matrix.png"
)

plt.show()


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "LSTM Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    BASE_DIR / "lstm_accuracy.png"
)

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = MODEL_DIR / "lstm_news_classifier.keras"

model.save(
    MODEL_FILE
)


# ============================================================
# SAVE TOKENIZER
# ============================================================

import pickle

TOKENIZER_FILE = MODEL_DIR / "lstm_tokenizer.pkl"

with open(
    TOKENIZER_FILE,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


print("\n========== SAVED FILES ==========")

print(
    "LSTM model:",
    MODEL_FILE
)

print(
    "Tokenizer:",
    TOKENIZER_FILE
)

print("\nLSTM training completed successfully!")