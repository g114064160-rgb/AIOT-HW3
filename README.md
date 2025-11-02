# AIOT-HW3 — Spam Classification (Phase1 baseline)

A small educational repository that demonstrates a simple ML workflow (CRISP-DM) and uses OpenSpec to track proposals and specs.

Contents
- `hw3/spam_baseline/` — Phase1 baseline code (data loader, TF-IDF + LinearSVC training, evaluation, Streamlit app).
- `openspec/` — OpenSpec conventions, proposals, and spec deltas for this project.

Tech stack
- Python 3.11+ (3.13 shown in this environment)
- numpy, pandas, scikit-learn, joblib, matplotlib
- streamlit (for demo UI)

Quick start (Windows PowerShell)

1) Create and activate a virtual environment, install deps

```powershell
cd 'c:\Users\user\Desktop\HW3\hw3\spam_baseline'
C:/Python313/python.exe -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Train the Phase1 baseline (downloads public CSV)

```powershell
python -m hw3.spam_baseline.cli --dataset-url "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv" --output-dir artifacts
```

After training you will find:
- `hw3/spam_baseline/artifacts/model.pkl` — trained pipeline
- `hw3/spam_baseline/artifacts/metrics.json` — evaluation metrics (accuracy, precision, recall, f1, confusion matrix)

3) Run the Streamlit demo

```powershell
cd 'c:\Users\user\Desktop\HW3\hw3\spam_baseline'
streamlit run streamlit_app.py
```

OpenSpec and proposals
- Project conventions and OpenSpec instructions are under `openspec/`.
- The spam classification work is tracked as OpenSpec changes:
  - `openspec/changes/add-spam-classification/` — umbrella change (Phase1 + Phase2 placeholders)
  - `openspec/changes/add-spam-classification-phase1-baseline/` — Phase1 baseline proposal

Notes and recommendations
- Artifacts (models, metrics) are currently stored in `hw3/spam_baseline/artifacts/`. For a cleaner repo, consider adding `hw3/spam_baseline/artifacts/` to `.gitignore` and using a storage backend or Git LFS for large files.
- If you want commits under your name/email instead of the local `hw3-bot` commit, set `git config user.name` and `user.email` locally and amend the commit.

Testing
- A couple of small tests were scaffolded under `hw3/spam_baseline/tests/` (pytest). Run tests from project root after activating your venv:

```powershell
cd 'c:\Users\user\Desktop\HW3'
pytest -q
```

Next steps (suggested)
- Phase2: feature engineering (n-grams, lexical features, embeddings)
- Phase2: hyperparameter tuning (CV + search)
- Phase2: deployable inference endpoint or Streamlit on Streamlit Cloud / Docker

If you want, I can:
- Remove artifacts from git and add `.gitignore` (recommended),
- Create a PR branch for the changes, or
- Add a Dockerfile and deployment instructions for Streamlit Cloud.
