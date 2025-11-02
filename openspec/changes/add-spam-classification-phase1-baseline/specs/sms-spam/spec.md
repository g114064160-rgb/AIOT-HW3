## ADDED Requirements

### Requirement: SMS Spam Classification - Phase1 Baseline
The system SHALL provide a reproducible baseline capability to train and evaluate an SMS spam classifier using a linear SVM over TF-IDF features.

#### Scenario: Train baseline model
- **GIVEN** a CSV dataset with two columns (label, message) available at the documented URL
- **WHEN** the user runs the training script with default parameters
- **THEN** the script SHALL produce a `metrics.json` containing accuracy, precision, recall, and F1-score and save a trained model artifact (pickle) to `artifacts/`.

#### Scenario: Evaluation metrics in acceptable ranges
- **GIVEN** the baseline trained model on the provided dataset
- **WHEN** evaluation completes
- **THEN** accuracy SHALL be >= 0.90 (baseline expectation) or the `metrics.json` SHALL clearly show measured values for comparison.
