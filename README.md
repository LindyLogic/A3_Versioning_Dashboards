
# A3 — Dataset/Model Versioning + Dashboards

420-976-VA Data Mining Project — Vanier College, Fall 2026 — Ryan Linder

Continues A1 (IMDB sentiment: TF-IDF + Logistic Regression) and A2 .
A3 adds **versioning** for the data and the model, an **MLflow-powered app**, and a **Grafana dashboard**.

## The story

| | Dataset v1 (original) | Dataset v2 (drifted) |
|---|---|---|
| What it is | A1's 50K IMDB reviews | Same reviews, "a year later on a phone app" |
| Length | ~119 words | ~29 words (15-50) |
| Words | normal | new slang: mid, fire, goated, cringe… + ~10% words missing |
| Positive share | 50% | 30% |

- **Model v1 (`a1_baseline`)** = the A1 recipe trained on v1 → 89.6% accuracy on v1, drops to 76.0% on drifted v2.
- **Model v2 (`drift_retrained`)** = same recipe + `class_weight="balanced"`, trained on v2 → 78.9% on v2.

## Architecture (`docker compose`)

| Service | What it does | URL |
|---|---|---|
| `postgres` | MLflow's database (runs, metrics, model registry) | — |
| `mlflow` | Tracking server + Model Registry + artifact store | http://localhost:5001 |
| `app` | Streamlit app: dataset/model selectors, predictions, charts | http://localhost:8501 |
| `grafana` | Dashboard, reads MLflow's Postgres database directly | http://localhost:3000 (admin/admin) |

## How to run

```bash
# 1. Put A1's cleaned data in data/imdb_processed.csv (not committed: too big)
# 2. Start everything
docker compose up -d --build
# 3. Step 1: build + log dataset v1 and v2
docker compose run --rm app python scripts/step1_dataset_versioning.py
# 4. Step 2: train + register model v1 and v2
docker compose run --rm app python scripts/step2_model_versioning.py
# 5. Open the app (8501), MLflow (5001) and Grafana (3000)
```

## Files
```
docker-compose.yml               # the 4 services
Dockerfile, requirements.txt     # one Python image for MLflow + the app
scripts/common.py                # shared names + A1 text cleaning
scripts/step1_dataset_versioning.py
scripts/step2_model_versioning.py
app/app.py                       # Streamlit app
grafana/provisioning/            # auto-connects Grafana to MLflow's database
grafana/dashboards/a3_dashboard.json
screenshots/                     # MLflow, app and Grafana screenshots
```






## AI Use Disclosure

I used Claude as a tutor and coding assistant for this assignment. It helped organise and explain me the concepts used of the assignment better then what I already knew great tool. 