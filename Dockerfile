# One Python image used by BOTH the MLflow server and the Streamlit app
FROM python:3.11-slim

WORKDIR /project

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK word lists for cleaning new reviews (same as A1/A2)
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab stopwords wordnet