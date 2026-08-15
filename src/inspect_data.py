import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

fake_path = DATA_DIR / "Fake.csv"
true_path = DATA_DIR / "True.csv"


# Load datasets
print("Loading Fake.csv...")
fake_df = pd.read_csv(fake_path)

print("Loading True.csv...")
true_df = pd.read_csv(true_path)


# Basic information
print("\n========== DATASET SHAPE ==========")
print("Fake news:", fake_df.shape)
print("True news:", true_df.shape)


# Columns
print("\n========== COLUMNS ==========")
print("Fake:", fake_df.columns.tolist())
print("True:", true_df.columns.tolist())


# First few rows
print("\n========== FAKE NEWS SAMPLE ==========")
print(fake_df.head())

print("\n========== TRUE NEWS SAMPLE ==========")
print(true_df.head())


# Missing values
print("\n========== MISSING VALUES ==========")
print("Fake:")
print(fake_df.isnull().sum())

print("\nTrue:")
print(true_df.isnull().sum())


# Duplicate rows
print("\n========== DUPLICATES ==========")
print("Fake duplicates:", fake_df.duplicated().sum())
print("True duplicates:", true_df.duplicated().sum())


# Dataset memory usage
print("\n========== MEMORY USAGE ==========")
print(
    f"Fake: {fake_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
)

print(
    f"True: {true_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
)