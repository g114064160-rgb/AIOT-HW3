## Why
We want a reproducible baseline for SMS spam classification so contributors can iterate on model improvements. A baseline SVM model trained on a small public SMS spam dataset gives a clear starting point for evaluation and future proposals (feature engineering, model selection, deployment).

## What Changes
- Add a new change `add-spam-classification-phase1-baseline` that introduces a baseline ML capability: training an SVM on the SMS spam dataset.
- Add data ingestion code and a small training script under `hw3/` or a clearly named directory.
- Add spec delta describing the new capability and an initial `tasks.md` with actionable steps.

## Impact
- Affected specs: `sms-spam` capability (new)
- Affected code: new scripts under `hw3/spam_baseline/` (loader, train, evaluate)
- Non-breaking: This is an additive change and doesn't modify existing specs.

## Data Source
- Public CSV: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv

## Note
This phase1 baseline is now referenced by the umbrella change `openspec/changes/add-spam-classification/` which groups Phase1 and Phase2 work. Keep this folder for history; implementers can work from either the umbrella change or this phase1 change.
