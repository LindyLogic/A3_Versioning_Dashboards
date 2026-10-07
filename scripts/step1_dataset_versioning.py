# ===== STEP 1 — DATASET VERSIONING =====
# Builds 2 versions of the IMDB dataset and logs each one to MLflow:
#   v1 = the original A1 data (50K reviews, 50/50 positive/negative)
#   v2 = a "drifted" copy, like reviews from a phone app a year later:
#        shorter, new slang words ("mid", "fire", "goated"), some words missing,
#        and way more negative reviews (70/30)
# Each version gets: the dataset itself (mlflow.log_input), metadata (params/tags),
# statistics (metrics), and the CSV file saved as an artifact.

import matplotlib
matplotlib.use("Agg")  # draw charts without a screen (we're inside Docker)
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (DATASET_EXPERIMENT, MLFLOW_URI, SOURCE_CSV, V1_CSV, V2_CSV)

SEED = 42
rng = np.random.default_rng(SEED)

# ---------- 1. Load A1's cleaned data and build v1 ----------
df = pd.read_csv(SOURCE_CSV)
df = df.dropna(subset=["processed_review"]).reset_index(drop=True)
df.insert(0, "review_id", range(len(df)))  # stable ID so v1 and v2 rows can be traced

# One fixed train/test split, shared by BOTH versions (85/15, stratified like A1).
# A review that is "test" in v1 is also "test" in v2, so no test data ever leaks into training.
train_ids, _ = train_test_split(df["review_id"], test_size=0.15,
                                stratify=df["label"], random_state=SEED)
df["split"] = np.where(df["review_id"].isin(train_ids), "train", "test")
v1 = df[["review_id", "processed_review", "label", "split"]]

# ---------- 2. Build v2 (the drifted version) ----------
# Drift 1 — class balance shifts to 70% negative / 30% positive
neg = v1[v1["label"] == 0]
pos = v1[v1["label"] == 1].sample(n=int(len(neg) * 30 / 70), random_state=SEED)
v2 = pd.concat([neg, pos]).sort_values("review_id").reset_index(drop=True)

# Drift 3 — new slang replaces common sentiment words (the v1 model has never seen these)
SLANG = {"great": "fire", "excellent": "fire", "good": "solid", "best": "goated", "amazing": "goated",
         "love": "stan", "loved": "stanned", "perfect": "chefkiss", "fun": "vibe", "beautiful": "stunning",
         "bad": "mid", "poor": "mid", "worst": "trash", "terrible": "trash", "awful": "cringe",
         "horrible": "cringe", "boring": "snoozefest", "dull": "snoozefest", "waste": "flop",
         "mess": "flop", "stupid": "dumb", "worse": "midder"}


def drift_text(text):
    words = text.split()[: rng.integers(15, 51)]     # Drift 2 — short reviews (15-50 words)
    words = [SLANG.get(w, w) for w in words]         # Drift 3 — slang
    kept = [w for w in words if rng.random() > 0.10] # Drift 4 — ~10% of words go missing
    return " ".join(kept) if kept else words[0]


v2["processed_review"] = v2["processed_review"].map(drift_text)

v1.to_csv(V1_CSV, index=False)
v2.to_csv(V2_CSV, index=False)


# ---------- 3. Statistics for each version ----------
def stats(d):
    n_words = d["processed_review"].str.split().str.len()
    vocab = set(" ".join(d["processed_review"]).split())
    return {
        "n_rows": len(d),
        "n_train": int((d["split"] == "train").sum()),
        "n_test": int((d["split"] == "test").sum()),
        "positive_rate": round(float(d["label"].mean()), 4),
        "avg_words": round(float(n_words.mean()), 2),
        "median_words": float(n_words.median()),
        "vocab_size": len(vocab),
    }, n_words, vocab


s1, len1, vocab1 = stats(v1)
s2, len2, vocab2 = stats(v2)


# ---------- 4. Drift indicators (v2 compared to v1) ----------
def psi(expected, actual, bins):
    """Population Stability Index: <0.1 no drift, 0.1-0.25 moderate, >0.25 big drift."""
    e = np.histogram(expected, bins=bins)[0] / len(expected) + 1e-4
    a = np.histogram(actual, bins=bins)[0] / len(actual) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))


drift = {
    "drift_length_psi": round(psi(len1, len2, bins=[0, 10, 20, 30, 50, 100, 200, 400, 5000]), 4),
    "drift_avg_words_change_pct": round((s2["avg_words"] - s1["avg_words"]) / s1["avg_words"] * 100, 2),
    "drift_positive_rate_shift": round(s2["positive_rate"] - s1["positive_rate"], 4),
    "drift_vocab_size_change_pct": round((s2["vocab_size"] - s1["vocab_size"]) / s1["vocab_size"] * 100, 2),
}

# Chart: review length, v1 vs v2
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(len1.clip(upper=300), bins=60, alpha=0.6, label="v1 (original)")
ax.hist(len2.clip(upper=300), bins=60, alpha=0.6, label="v2 (drifted)")
ax.set_xlabel("Words per review (after cleaning)")
ax.set_ylabel("Number of reviews")
ax.set_title("Review length drift: v1 vs v2")
ax.legend()
fig.tight_layout()
fig.savefig("data/length_drift.png", dpi=120)

# ---------- 5. Log both versions to MLflow ----------
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(DATASET_EXPERIMENT)

versions = [
    ("v1", v1, V1_CSV, s1, {}, "Original IMDB 50K reviews from A1 (cleaned + lemmatized)."),
    ("v2", v2, V2_CSV, s2, drift, "Drifted copy of v1: 15-50 word reviews, new slang words, ~10% words dropped, 70/30 negative/positive."),
]

for version, data, path, s, d, description in versions:
    with mlflow.start_run(run_name=f"dataset-{version}"):
        # The dataset itself -> shows in the MLflow UI "Datasets used" column, with a digest (fingerprint)
        ds = mlflow.data.from_pandas(data, source=path, name="imdb_reviews", targets="label")
        mlflow.log_input(ds, context=version)

        # Metadata
        mlflow.set_tags({"dataset_name": "imdb_reviews", "dataset_version": version,
                         "mlflow.note.content": description})
        mlflow.log_params({"dataset_version": version, "source": "Kaggle IMDB 50K (A1)",
                           "split": "85/15 stratified, seed 42", "digest": ds.digest,
                           "drift_recipe": "none" if version == "v1" else "truncate 15-50 words, slang swap, drop 10% words, 70/30 neg/pos"})

        # Statistics + drift indicators
        mlflow.log_metrics({**s, **d})

        # The actual files
        mlflow.log_artifact(path, artifact_path="data")
        if version == "v2":
            mlflow.log_artifact("data/length_drift.png", artifact_path="charts")

    print(f"Logged dataset {version}: {s}")

print("Drift indicators (v2 vs v1):", drift)