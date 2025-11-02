Spam classification Phase1 baseline

This package contains a minimal baseline pipeline to train a TF-IDF + Linear SVM classifier on the public SMS spam dataset.

Quick start

1. Install dependencies (prefer a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run training (uses the public raw CSV URL):

```powershell
python -m hw3.spam_baseline.cli --dataset-url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv" --output-dir artifacts
```

Artifacts
- `artifacts/model.pkl` - trained pipeline
- `artifacts/metrics.json` - evaluation metrics

Notes
- This is a Phase1 baseline. Phase2 tasks (feature engineering, hyperparameter tuning, deployment) are planned in OpenSpec changes.
