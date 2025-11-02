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

# A few example messages for quick testing/demo
SAMPLE_MESSAGES = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive a entry question(std txt rate)",
    "Hey mate, are we still meeting for coffee tomorrow?",
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
    "Can you send me the report by EOD?",
]


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
    # Simple page styling
    st.markdown("<h1 style='text-align:center'>SMS Spam Classifier <small style=\"font-size:14px;color:gray\">(Phase1 baseline)</small></h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Model & Data")
        dataset_url = st.text_input("Dataset raw CSV URL", DEFAULT_DATASET_URL)
        st.caption("Dataset: SMS spam dataset (public)")
        if st.button("Train baseline model now"):
            # training may take a while; show a simple progress indicator
            with st.spinner("Training model — this may take a minute..."):
                try:
                    train_from_url(dataset_url, output_dir=ARTIFACTS_DIR)
                    st.success("Training finished — model saved to artifacts/")
                    # clear cached model
                    load_model.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        st.markdown("---")
        st.subheader("Quick examples")
        example = st.selectbox("Pick a sample message", options=["(none)"] + SAMPLE_MESSAGES)
        if st.button("Fill example into message box") and example != "(none)":
            # communicate via session_state to keep behavior simple
            st.session_state._sample_text = example
        st.markdown("---")
        # model download
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as mf:
                model_bytes = mf.read()
            st.download_button("Download model.pkl", data=model_bytes, file_name="model.pkl")

    st.markdown("Enter an SMS message below and click Predict to see the baseline model's output.")
    # support filling from sidebar example
    initial = st.session_state.get("_sample_text", "")
    input_text = st.text_area("Message", value=initial, height=150)

    model = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)

    if metrics is not None:
        # show metric cards
        st.markdown("### Latest evaluation metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        mcol2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        mcol3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        mcol4.metric("F1", f"{metrics.get('f1', 0):.3f}")

        # confusion matrix plot
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            cm = metrics.get("confusion_matrix")
            if cm is not None:
                cm_arr = np.array(cm)
                fig, ax = plt.subplots(figsize=(3, 3))
                im = ax.imshow(cm_arr, cmap="Blues")
                for (i, j), val in np.ndenumerate(cm_arr):
                    ax.text(j, i, int(val), ha="center", va="center", color="black")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["ham", "spam"])
                ax.set_yticklabels(["ham", "spam"])
                st.pyplot(fig)
        except Exception:
            # fallback: show raw JSON
            with st.expander("Raw metrics"):
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
                    st.markdown(f"### Prediction: **{label.upper()}**")
                    if conf is not None:
                        st.progress(min(max(conf, 0.0), 1.0))
                        st.write(f"Confidence (pseudo): {conf:.2f}")
    


if __name__ == "__main__":
    main()
