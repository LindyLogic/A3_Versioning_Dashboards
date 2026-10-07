# ===== STEP 2 — MODEL VERSIONING =====
# Trains the A1 model recipe (TF-IDF + Logistic Regression) twice and registers
# both in the MLflow Model Registry under one name, "imdb-sentiment":
#   version 1 = trained on dataset v1  -> this is the A1 model
#   version 2 = trained on dataset v2  -> retrained on the drifted data
# Every model is tested on BOTH test sets, so we can compare them fairly.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from common import MLFLOW_URI, MODEL_EXPERIMENT, MODEL_NAME, V1_CSV, V2_CSV

# Exact A1 settings (from A1 notebook)
TFIDF_PARAMS = dict(max_features=20000, ngram_range=(1, 1), min_df=2)
LOGREG_PARAMS = dict(C=1.0, max_iter=1000, solver="liblinear", random_state=42)

data = {"v1": pd.read_csv(V1_CSV), "v2": pd.read_csv(V2_CSV)}
for d in data.values():
    d["processed_review"] = d["processed_review"].fillna("")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MODEL_EXPERIMENT)
client = MlflowClient()


def evaluate(model, df):
    test = df[df["split"] == "test"]
    pred = model.predict(test["processed_review"])
    proba = model.predict_proba(test["processed_review"])[:, 1]
    y = test["label"]
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, test, pred


# v2 gets ONE change: class_weight="balanced", because v2 is 70% negative.
# Without it, the model leans toward "negative" and misses positive reviews.
plan = [
    ("v1", "a1_baseline", None,
     "A1 model: TF-IDF + Logistic Regression trained on the original IMDB data (v1)."),
    ("v2", "drift_retrained", "balanced",
     "A1 recipe retrained on the drifted dataset (v2) with class_weight=balanced for the 70/30 imbalance."),
]

for train_version, alias, class_weight, description in plan:
    train_df = data[train_version]
    train = train_df[train_df["split"] == "train"]

    with mlflow.start_run(run_name=f"model-trained-on-{train_version}") as run:
        # Which dataset version trained this model (links model <-> dataset in the UI)
        mlflow.log_input(mlflow.data.from_pandas(train_df, source=f"data/imdb_{train_version}.csv",
                                                 name="imdb_reviews", targets="label"),
                         context="training")
        mlflow.set_tags({"dataset_version": train_version, "mlflow.note.content": description})
        mlflow.log_params({"trained_on": train_version, "n_train": len(train),
                           "logreg_class_weight": str(class_weight),
                           **{f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()},
                           **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()}})

        # Train: the pipeline holds the vectorizer AND the model, so they're versioned together
        model = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                          ("logreg", LogisticRegression(**LOGREG_PARAMS, class_weight=class_weight))])
        model.fit(train["processed_review"], train["label"])

        # Test on BOTH dataset versions
        for test_version, test_df in data.items():
            m, test, pred = evaluate(model, test_df)
            mlflow.log_metrics({f"{k}_on_{test_version}_test": round(v, 4) for k, v in m.items()})
            if test_version == train_version:
                mlflow.log_metrics({k: round(v, 4) for k, v in m.items()})  # headline metrics
                fig, ax = plt.subplots(figsize=(4, 4))
                ConfusionMatrixDisplay.from_predictions(test["label"], pred, ax=ax, colorbar=False,
                                                        display_labels=["negative", "positive"])
                ax.set_title(f"Model trained on {train_version} (tested on {test_version})")
                fig.tight_layout()
                mlflow.log_figure(fig, "charts/confusion_matrix.png")
                plt.close(fig)

        # Save + register the model -> creates the next version of "imdb-sentiment"
        sample = train["processed_review"].head(5)
        info = mlflow.sklearn.log_model(model, artifact_path="model",
                                        signature=infer_signature(sample, model.predict(sample)),
                                        input_example=sample.tolist(),
                                        registered_model_name=MODEL_NAME)

    mv = client.get_model_version(MODEL_NAME, info.registered_model_version)
    client.update_model_version(MODEL_NAME, mv.version, description=description)
    client.set_model_version_tag(MODEL_NAME, mv.version, "dataset_version", train_version)
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    print(f"Registered {MODEL_NAME} version {mv.version} (alias '{alias}'), trained on {train_version}")

client.update_registered_model(MODEL_NAME, description="IMDB review sentiment (positive/negative). TF-IDF + Logistic Regression from A1.")