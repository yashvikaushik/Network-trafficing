# AI data and network-state pipeline

This area is for AI Member 1's technical data pipeline only.

## Scope

- Dataset inspection and analysis, only after explicit authorization
- Data preprocessing and validation
- Network and flow feature extraction
- Time-window creation and network-state construction
- Technical label/target preparation
- Temporal sequence generation
- Leakage-safe train/validation/test splitting
- ML-ready dataset creation, documentation, and tests

## Explicitly out of scope

Do not implement forecasting models, LSTM, GRU, Transformer, XAI, MITRE ATT&CK mappings, blockchain, backend, or frontend work here.

## Layout

- `data/`: local datasets, separated by raw, interim, and processed lifecycle stage. Contents are ignored by Git.
- `src/`: future pipeline modules, organized by pipeline responsibility.
- `configs/`: future declarative pipeline configuration.
- `tests/`: future validation and pipeline tests.
- `outputs/`: generated states, sequences, and reports; ignored by Git.
- `notebooks/`: future exploration notebooks after dataset-analysis approval.

No dataset has been inspected, downloaded, altered, or assumed during scaffold creation.
