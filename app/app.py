# ===== STEP 3 — THE MLFLOW APP (Streamlit) =====
# A web page that talks to MLflow and lets you:
#   - pick a DATASET version and see its metadata, stats and history
#   - pick a MODEL version and see its metrics
#   - type a review and get a prediction
#   - compare all model versions in charts

import sys

sys.path.append("scripts")  # so we can reuse common.py

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

from common import DATASET_EXPERIMENT, MLFLOW_URI, MODEL_NAME, preprocess

st.set_page_config(page_title="IMDB Sentiment — Versions", layout="wide")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


# ---------- Load info from MLflow (cached so the page stays fast) ----------
@st.cache_data(ttl=60)
def dataset_runs():
    runs = mlflow.search_runs(experiment_names=[DATASET_EXPERIMENT], order_by=["start_time DESC"])
    return runs.drop_duplicates("tags.dataset_version")  # newest run of each version


@st.cache_data(ttl=60)
def model_versions():
    rows = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        mv = client.get_model_version(MODEL_NAME, mv.version)  # full details incl. aliases
        run = client.get_run(mv.run_id)
        rows.append({"version": int(mv.version), "aliases": ", ".join(mv.aliases),
                     "trained_on": run.data.params.get("trained_on"),
                     "description": mv.description, "run_id": mv.run_id, **run.data.metrics})
    return pd.DataFrame(rows).sort_values("version")


@st.cache_resource
def load_model(version):
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


st.title("🎬 IMDB Sentiment — Dataset & Model Versions")
st.caption(f"Connected to MLflow at {MLFLOW_URI}")

ds = dataset_runs()
mv = model_versions()
if ds.empty or mv.empty:
    st.error("No data in MLflow yet. Run step1_dataset_versioning.py and step2_model_versioning.py first.")
    st.stop()

# ---------- Sidebar: the two selectors ----------
st.sidebar.header("Pick versions")
ds_version = st.sidebar.selectbox("Dataset version", sorted(ds["tags.dataset_version"]))
model_version = st.sidebar.selectbox(
    "Model version", mv["version"].tolist(), index=len(mv) - 1,
    format_func=lambda v: f"v{v} — {mv.set_index('version').loc[v, 'aliases']}")

page = st.sidebar.radio("Section", ["📁 Dataset", "🤖 Model", "✍️ Predict", "📊 Compare models"])

# ---------- Section 1: dataset version ----------
if page == "📁 Dataset":
    row = ds[ds["tags.dataset_version"] == ds_version].iloc[0]
    st.subheader(f"Dataset {ds_version}")
    st.write(row.get("tags.mlflow.note.content", ""))
    c = st.columns(4)
    c[0].metric("Rows", f"{int(row['metrics.n_rows']):,}")
    c[1].metric("Positive rate", f"{row['metrics.positive_rate']:.0%}")
    c[2].metric("Avg words / review", f"{row['metrics.avg_words']:.0f}")
    c[3].metric("Vocabulary size", f"{int(row['metrics.vocab_size']):,}")

    st.markdown("**Metadata**")
    st.json({k.replace("params.", ""): v for k, v in row.items() if k.startswith("params.")})

    if ds_version != "v1":
        st.markdown("**Drift indicators vs v1**")
        d = st.columns(4)
        d[0].metric("Length PSI", f"{row['metrics.drift_length_psi']:.2f}", help=">0.25 = big drift")
        d[1].metric("Avg length change", f"{row['metrics.drift_avg_words_change_pct']:.0f}%")
        d[2].metric("Positive rate shift", f"{row['metrics.drift_positive_rate_shift'] * 100:+.0f} pts")
        d[3].metric("Vocab size change", f"{row['metrics.drift_vocab_size_change_pct']:.0f}%")

    st.markdown("**Version history** (every logged dataset run)")
    history = mlflow.search_runs(experiment_names=[DATASET_EXPERIMENT], order_by=["start_time DESC"])
    st.dataframe(history[["tags.dataset_version", "params.digest", "metrics.n_rows",
                          "metrics.positive_rate", "metrics.avg_words", "start_time"]]
                 .rename(columns=lambda c: c.split(".")[-1]), hide_index=True)

# ---------- Section 2: model version + metrics ----------
if page == "🤖 Model":
    m = mv.set_index("version").loc[model_version]
    st.subheader(f"Model {MODEL_NAME} v{model_version} ({m['aliases']})")
    st.write(m["description"])
    st.write(f"Trained on dataset **{m['trained_on']}**")
    metrics = pd.DataFrame({
        "v1 test (original)": [m[f"{k}_on_v1_test"] for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]],
        "v2 test (drifted)": [m[f"{k}_on_v2_test"] for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]],
    }, index=["accuracy", "f1", "precision", "recall", "roc_auc"])
    st.bar_chart(metrics, stack=False)
    st.dataframe(metrics.round(3))
    img = mlflow.artifacts.download_artifacts(run_id=m["run_id"], artifact_path="charts/confusion_matrix.png")
    st.image(img, caption="Confusion matrix on its own test set", width=380)

# ---------- Section 3: prediction ----------
if page == "✍️ Predict":
    st.subheader("Try a review")
    text = st.text_area("Review", "Honestly this movie was fire. The cast was goated and the ending was chefkiss.")
    if st.button("Predict with both models"):
        clean = preprocess(text)
        cols = st.columns(len(mv))
        for col, v in zip(cols, mv["version"]):
            p = load_model(v).predict_proba([clean])[0, 1]
            label = "positive 😀" if p >= 0.5 else "negative 😞"
            star = " ⬅ selected" if v == model_version else ""
            col.metric(f"Model v{v}{star}", label, f"{p:.0%} positive", delta_color="off")
        st.caption(f"Cleaned text the model sees: `{clean}`")

# ---------- Section 4: compare all model versions ----------
if page == "📊 Compare models":
    st.subheader("Model comparison")
    for metric in ["accuracy", "f1"]:
        chart = mv.set_index(mv["version"].map(lambda v: f"model v{v}"))[
            [f"{metric}_on_v1_test", f"{metric}_on_v2_test"]]
        chart.columns = ["on v1 test (original)", "on v2 test (drifted)"]
        st.markdown(f"**{metric.upper()}**")
        st.bar_chart(chart, stack=False)
    st.dataframe(mv.drop(columns=["run_id", "description"]).set_index("version").round(3))
    