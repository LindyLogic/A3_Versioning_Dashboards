# ===== Shared settings + text cleaning, used by every script and the app =====
import os
import re
import string

# Where the MLflow server lives (inside Docker it's the "mlflow" container)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Names we reuse everywhere, so they always match
DATASET_EXPERIMENT = "imdb-dataset-versions"   # Step 1: one run per dataset version
MODEL_EXPERIMENT = "imdb-model-training"       # Step 2: one run per model training
MODEL_NAME = "imdb-sentiment"                  # name in the MLflow Model Registry

DATA_DIR = os.getenv("DATA_DIR", "data")
SOURCE_CSV = os.path.join(DATA_DIR, "imdb_processed.csv")   # cleaned data from A1
V1_CSV = os.path.join(DATA_DIR, "imdb_v1.csv")
V2_CSV = os.path.join(DATA_DIR, "imdb_v2_drifted.csv")

# ----- Same cleaning steps as A1 (so new reviews look like the training data) -----
HTML_TAG_RE = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation + string.digits)


def clean_text(text):
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text):
    """Raw review -> cleaned, tokenized, lemmatized text (A1 recipe)."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    tokens = word_tokenize(clean_text(text))
    return " ".join(lem.lemmatize(t) for t in tokens if t not in stop and len(t) > 1)