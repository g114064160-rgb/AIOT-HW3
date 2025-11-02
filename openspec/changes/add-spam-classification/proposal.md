## Why
We want one umbrella change that groups the Phase1 baseline and planned Phase2 work (feature engineering, hyperparameter tuning, deployment) so reviewers and implementers can see the roadmap in a single place.

## What Changes
- Consolidate Phase1 baseline (SVM TF-IDF) and Phase2 placeholders into a single change `add-spam-classification` that documents the end-to-end plan and per-phase tasks.
- Provide spec deltas for the baseline capability and mark Phase2 items as planned deltas (placeholders).

## Impact
- This is additive: it references existing artifacts and provides a single entry-point for the spam classification effort.
- Affected specs: `sms-spam` (phase1 baseline), future deltas for feature-engineering, hyperparameter-tuning, and deployment.

## Phases
- Phase1: Baseline — TF-IDF + Linear SVM training/evaluation (existing change `add-spam-classification-phase1-baseline`).
- Phase2: Feature engineering, hyperparameter tuning, deployment — placeholders exist under `openspec/changes/add-spam-classification-phase2-*`.

## Notes
- This umbrella change does not remove the phase-specific proposal directories; they remain for iteration history. Implementers may prefer to work from this umbrella change branch `feature/add-spam-classification`.
