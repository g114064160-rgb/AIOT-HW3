## 1. Implementation
- [ ] 1.1 Create data loader: `hw3/spam_baseline/data.py` that downloads and parses the CSV into a train/test split.
- [ ] 1.2 Implement training script: `hw3/spam_baseline/train.py` that trains an SVM baseline (TF-IDF features + linear SVM).
- [ ] 1.3 Implement evaluation: output accuracy, precision, recall, F1, confusion matrix; save a small `metrics.json`.
- [ ] 1.4 Add a minimal CLI or `main.py` to run training and evaluation locally.
- [ ] 1.5 Add unit tests: data loader smoke test, training smoke test (runs on a tiny subset).
- [ ] 1.6 Add `openspec` spec delta under `specs/sms-spam/spec.md`.
- [ ] 1.7 Validate proposal formatting and spec scenarios.

## 2. Acceptance
- [ ] All tasks above completed and tests pass locally.
- [ ] `openspec validate add-spam-classification-phase1-baseline --strict` shows no blocking errors.
