import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = BASE_DIR / "data" / "processed_news_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading processed dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# Remove possible missing values
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"]
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot()

plt.title("TF-IDF + Logistic Regression")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "tfidf_confusion_matrix.png"
)

plt.show()


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

print("\nSaving model...")

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.pkl"
)


print("\nModel saved successfully!")

print(
    "Vectorizer:",
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print(
    "Classifier:",
    MODEL_DIR / "logistic_regression.pkl"
)