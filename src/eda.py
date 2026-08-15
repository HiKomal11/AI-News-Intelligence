import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed_news_dataset.csv"

# Load dataset
print("Loading processed dataset...")
df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# 1. CLASS DISTRIBUTION
# --------------------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

class_counts = df["label"].value_counts()

print(class_counts)

print("\nFake:", class_counts.get(0, 0))
print("Real:", class_counts.get(1, 0))


# --------------------------------------------------
# 2. ARTICLE LENGTH
# --------------------------------------------------

df["word_count"] = df["clean_text"].str.split().str.len()

print("\n========== ARTICLE LENGTH ==========")

print(df["word_count"].describe())


# --------------------------------------------------
# 3. SUBJECT DISTRIBUTION
# --------------------------------------------------

print("\n========== TOP SUBJECTS ==========")

print(df["subject"].value_counts().head(15))


# --------------------------------------------------
# 4. MOST COMMON WORDS
# --------------------------------------------------

print("\n========== MOST COMMON WORDS ==========")

all_words = " ".join(df["clean_text"].dropna()).split()

word_counts = pd.Series(all_words).value_counts()

print(word_counts.head(30))


# --------------------------------------------------
# 5. FAKE NEWS WORDS
# --------------------------------------------------

print("\n========== COMMON WORDS IN FAKE NEWS ==========")

fake_words = " ".join(
    df[df["label"] == 0]["clean_text"].dropna()
).split()

fake_word_counts = pd.Series(fake_words).value_counts()

print(fake_word_counts.head(20))


# --------------------------------------------------
# 6. REAL NEWS WORDS
# --------------------------------------------------

print("\n========== COMMON WORDS IN REAL NEWS ==========")

real_words = " ".join(
    df[df["label"] == 1]["clean_text"].dropna()
).split()

real_word_counts = pd.Series(real_words).value_counts()

print(real_word_counts.head(20))


# --------------------------------------------------
# 7. VISUALIZATION - CLASS DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["label"].value_counts().sort_index().plot(kind="bar")

plt.title("Fake vs Real News Distribution")
plt.xlabel("Label (0 = Fake, 1 = Real)")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(BASE_DIR / "class_distribution.png")

plt.show()


# --------------------------------------------------
# 8. VISUALIZATION - ARTICLE LENGTH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["word_count"],
    bins=50
)

plt.title("Article Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(BASE_DIR / "article_length_distribution.png")

plt.show()


print("\nEDA completed successfully.")