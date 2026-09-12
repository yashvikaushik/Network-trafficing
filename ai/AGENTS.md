# AI Member 1 workspace instructions

## Ownership and scope

This `ai/` workspace supports only the data and network-state pipeline: inspection, preprocessing, validation, feature extraction, time windows, state construction, technical target preparation, temporal sequences, leakage-safe splitting, ML-ready dataset creation, documentation, and tests.

## Current gate

Do not inspect, download, modify, move, or infer properties of any dataset until the user explicitly authorizes dataset analysis. Treat `data/raw/` as immutable once data is supplied.

## Out of scope

Do not implement forecasting models or LSTM, GRU, Transformer, XAI, MITRE ATT&CK, blockchain, backend, or frontend components.

## Data safety and reproducibility

- Keep datasets and generated outputs out of Git.
- Preserve source data and record lineage for each derived artifact.
- Validate schemas, types, timestamps, missingness, duplicates, and ordering before transformations.
- Define time windows and labels using only information available at the prediction cutoff.
- Split chronologically before fitting learned preprocessing, feature-selection, or normalization steps.
- Fit such transformations on training data only, then apply them unchanged to validation and test data.
- Keep stable entity/flow identifiers only where needed for grouping and leakage checks; exclude identifiers that would leak outcomes from model-ready features.

## Engineering expectations

- Keep pipeline logic modular under `src/` and configurations declarative under `configs/`.
- Add tests for validation rules, boundary conditions, chronological splits, and leakage prevention.
- Write outputs to `outputs/` and document artifact schemas and assumptions in `outputs/reports/` or project documentation.
- Do not add implementation code until a separately approved task requests it.
