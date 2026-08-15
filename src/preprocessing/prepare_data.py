import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# Load datasets
print("Loading datasets...")

fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")


# Add labels
fake_df["label"] = 0
true_df["label"] = 1


# Combine title and article text
fake_df["combined_text"] = (
    fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
)

true_df["combined_text"] = (
    true_df["title"].fillna("") + " " + true_df["text"].fillna("")
)


# Keep only the columns needed for classification
fake_df = fake_df[["title", "text", "combined_text", "subject", "date", "label"]]
true_df = true_df[["title", "text", "combined_text", "subject", "date", "label"]]


# Combine Fake and True datasets
df = pd.concat([fake_df, true_df], ignore_index=True)


print("\nBefore removing duplicates:")
print("Total records:", len(df))


# Remove duplicate articles
df = df.drop_duplicates(subset=["combined_text"])


print("After removing duplicates:")
print("Total records:", len(df))


# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Display class distribution
print("\n========== CLASS DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\nFake:", (df["label"] == 0).sum())
print("Real:", (df["label"] == 1).sum())


# Save cleaned dataset
output_path = DATA_DIR / "news_dataset.csv"
df.to_csv(output_path, index=False)


print("\n========== FINAL DATASET ==========")
print("Shape:", df.shape)
print("Saved to:", output_path)