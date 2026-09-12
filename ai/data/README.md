# Data directories

Place local data only after explicit authorization to begin dataset analysis.

- `raw/`: immutable source files. Never edit these in place.
- `interim/`: reproducible intermediate artifacts.
- `processed/`: validated ML-ready technical datasets.

All data contents are Git-ignored to prevent accidental commits. The tracked `.gitkeep` files preserve the directory structure only.
