import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

label_file = RESULTS_DIR / "bigdata_label_distribution.csv"

label_df = pd.read_csv(label_file)

label_names = {
    0: "Fake",
    1: "Real"
}

label_df["label_name"] = label_df["label"].map(label_names)


plt.figure(figsize=(7, 5))

plt.bar(
    label_df["label_name"],
    label_df["article_count"]
)

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "bigdata_label_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

subject_file = (
    RESULTS_DIR / "bigdata_subject_distribution.csv"
)

subject_df = pd.read_csv(subject_file)

plt.figure(figsize=(10, 6))

plt.barh(
    subject_df["subject"],
    subject_df["article_count"]
)

plt.title("News Articles by Subject")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "bigdata_subject_distribution.png",
    dpi=300
)

plt.close()


print("======================================")
print("    BIG DATA VISUALIZATIONS SAVED")
print("======================================")

print(
    RESULTS_DIR /
    "bigdata_label_distribution.png"
)

print(
    RESULTS_DIR /
    "bigdata_subject_distribution.png"
)