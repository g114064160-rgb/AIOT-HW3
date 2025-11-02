## Project Context

### Purpose
This repository contains examples and exercises for simple machine learning workflows (CRISP-DM) and small demo apps. The immediate goals are:
- Provide reproducible, educational examples for building and evaluating simple ML models (linear regression, spam classification baseline).
- Capture requirements and proposals as OpenSpec change proposals under `openspec/changes/`.
- Keep `openspec/specs/` as the source of truth for capabilities that are implemented and validated.

### Tech Stack
- Language: Python 3.x (examples use `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `streamlit`).
- Notebooks / scripts: plain Python scripts and small Streamlit apps.
- Testing: lightweight unit tests (pytest recommended) for model training/pipeline code.
- Packaging / environment: `requirements.txt` for Python dependencies.

### Project Conventions

#### Code Style
- Use idiomatic Python (PEP8). Keep functions small and focused.
- Use type hints for public functions where helpful.
- Keep reproducible scripts under top-level directories (e.g., `hw3/`, `L5/`, `HW1/`).

#### Architecture Patterns
- Small, single-responsibility scripts and modules. Start simple; extract utilities when multiple examples reuse code.
- ML experiments follow a lightweight pipeline: data acquisition -> preprocessing -> training -> evaluation -> (optionally) deployment.

#### Testing Strategy
- Add unit tests for deterministic pieces (data loader, preprocessing, evaluation metrics). For model training, include a smoke test that runs on a tiny dataset and checks shapes and metric ranges.
- Use `pytest` and keep tests fast (< 1s) where possible.

#### Git Workflow
- Branching: `main` is the canonical branch. Create short-lived feature branches named `feature/<change-id>` or `changes/<change-id>` when implementing proposals.
- Commits: small, focused commits. Use conventional commit-style messages (e.g., `feat: add initial spam dataset loader`).

### Domain Context
- Example capabilities in this repo are machine learning focused (regression, classification). Data is small, synthetic, or public CSV datasets.
- The spam classification proposal will use the public SMS spam CSV dataset and focus on a baseline SVM classifier (phase1-baseline).

### Important Constraints
- Avoid large datasets or heavyweight infra in the examples. Keep examples runnable on a developer laptop.
- Do not include secrets or external credentials in the repo.

### External Dependencies
- Public dataset: SMS spam CSV from Packt repo (CSV URL can be referenced in `changes/*/proposal.md`).
- Python packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `streamlit` (optional for demos), `pytest` for tests.

### How the assistant will help
- I will author OpenSpec proposals under `openspec/changes/` and add spec deltas under `openspec/changes/<change-id>/specs/`.
- I will create `proposal.md`, `tasks.md`, and minimal `design.md` when appropriate and run local validation guidance steps.

## Minimal Contact Points
- `openspec/project.md` - this file (conventions & context)
- `openspec/specs/` - implemented capability specs
- `openspec/changes/` - active proposals
