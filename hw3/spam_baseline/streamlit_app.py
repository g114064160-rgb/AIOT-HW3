import os
import sys
import json
import time
from typing import Optional

# Ensure repo root is on sys.path so imports like `hw3.spam_baseline` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import joblib

try:
    from hw3.spam_baseline.train import train_from_url, build_pipeline
except Exception:
    # Fallback: load the train module directly from file path (handles some deployment import quirks)
    import importlib.util

    _train_path = os.path.join(os.path.dirname(__file__), "train.py")
    if os.path.exists(_train_path):
        spec = importlib.util.spec_from_file_location("spam_baseline.train", _train_path)
        _train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_train_mod)
        train_from_url = getattr(_train_mod, "train_from_url")
        build_pipeline = getattr(_train_mod, "build_pipeline")
    else:
        raise

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "master/Chapter03/datasets/sms_spam_no_header.csv"
)


@st.cache_resource
def load_model(path: str) -> Optional[object]:
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def load_metrics(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def predict_text(model, text: str):
    if text is None or text.strip() == "":
        return None, None
    pred = model.predict([text])[0]
    # Decision function -> pseudo-confidence
    score = None
    if hasattr(model, "decision_function"):
        try:
            val = float(model.decision_function([text])[0])
            # convert to 0..1 via sigmoid for display
            import math

            score = 1.0 / (1.0 + math.exp(-val))
        except Exception:
            score = None
    return pred, score


def main():
    st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
    st.title("SMS Spam Classifier (Phase1 baseline)")

    with st.sidebar:
        st.header("Model & Data")
        dataset_url = st.text_input("Dataset raw CSV URL", DEFAULT_DATASET_URL)
        if st.button("Train baseline model now"):
            with st.spinner("Training model — this may take a minute..."):
                try:
                    train_from_url(dataset_url, output_dir=ARTIFACTS_DIR)
                    st.success("Training finished — model saved to artifacts/")
                    # clear cached model
                    load_model.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("Enter an SMS message below and click Predict to see the baseline model's output.")
    input_text = st.text_area("Message", height=150)

    model = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)

    if metrics is not None:
        with st.expander("Latest evaluation metrics"):
            st.json(metrics)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Predict"):
            if model is None:
                st.warning("No trained model found. Please train the baseline using the sidebar button.")
            else:
                pred, conf = predict_text(model, input_text)
                if pred is None:
                    st.info("Please enter text to classify.")
                else:
                    label = "spam" if int(pred) == 1 else "ham"
                    st.subheader(f"Prediction: {label}")
                    if conf is not None:
                        st.write(f"Confidence (pseudo): {conf:.2f}")
    with col2:
        st.write("Model")
        if model is None:
            st.write("No model loaded")
        else:
            st.write(type(model))

    st.markdown("---")
    st.markdown("This app uses a TF-IDF vectorizer + LinearSVC as a Phase1 baseline. It's intended for demo and educational purposes.")


if __name__ == "__main__":
    main()
