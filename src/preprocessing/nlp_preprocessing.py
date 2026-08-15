import re
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "news_dataset.csv"
OUTPUT_FILE = DATA_DIR / "processed_news_dataset.csv"


# NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and normalize news article text.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


print("\nStarting NLP preprocessing...")

df["clean_text"] = df["combined_text"].apply(clean_text)


# Remove empty records after preprocessing
df = df[df["clean_text"].str.strip() != ""]


print("\nPreprocessing completed.")

print("Final dataset shape:", df.shape)


# Display examples
print("\n========== BEFORE / AFTER ==========")

for i in range(min(3, len(df))):
    print("\nOriginal:")
    print(df.iloc[i]["combined_text"][:300])

    print("\nCleaned:")
    print(df.iloc[i]["clean_text"][:300])


# Save processed dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)
