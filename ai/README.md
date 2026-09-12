## Current ML Baseline — Option A

### Status

The first end-to-end temporal forecasting baseline has been implemented and trained using the CSE-CIC-IDS2018 processed traffic dataset.

The current baseline is intentionally binary:

- `BENIGN = 0`
- `ATTACK = 1`

The purpose of this stage is to establish a reproducible temporal ML baseline before adding XAI, MITRE ATT&CK mapping, blockchain integrity, and the packet-level Option-B pipeline.

### Data Pipeline

The current Option-A pipeline performs:

1. Streaming/chunked inspection of the CSE-CIC-IDS2018 CSV files.
2. Removal of invalid/non-finite records and embedded repeated headers.
3. Temporal ordering within source-day/file segments.
4. Construction of one-minute network states.
5. Aggregation into 21 numerical network-state features.
6. Construction of temporal sequences using the previous 10 states.
7. Binary target conversion:
   - `Benign → BENIGN`
   - every non-Benign source label → `ATTACK`
8. Chronological train/validation/test splitting.

The pipeline processes all downloaded processed CSE-CIC-IDS2018 CSV files rather than training on a single CSV.

### Dataset Processing Results

- Source records read: `16,233,002`
- Removed records: `95,833`
- Network states: `4,987`
- Temporal sequences: `4,887`
- Features per state: `21`
- Sequence length: `10`

### Chronological Split

The dataset is split by complete source days/files rather than randomly.

| Split | Samples | Source period |
|---|---:|---|
| Train | 3,247 | 14, 15, 16, 20, 21, 22, 23 Feb 2018 |
| Validation | 561 | 28 Feb 2018 |
| Test | 1,079 | 1–2 Mar 2018 |

No sequence crosses a source-day/file boundary.

### Binary Class Distribution

| Split | BENIGN | ATTACK |
|---|---:|---:|
| Train | 2,557 | 690 |
| Validation | 424 | 137 |
| Test | 588 | 491 |

### Temporal LSTM

The baseline model uses:

- PyTorch LSTM
- Input size: 21
- Sequence length: 10
- Hidden size: 64
- LSTM layers: 2
- Dropout: 0.2
- Output classes: 2
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Class-weighted cross entropy
- Early stopping
- Random seed: 42

The model was trained using Apple Metal Performance Shaders (MPS) on an Apple Silicon Mac.

### Leakage Controls

Several safeguards are used:

- Train/validation/test are separated chronologically.
- Random row-level splitting is not used.
- The test period is not used for model selection.
- Feature scaling is fitted only on the training set.
- Sequence targets correspond to the state immediately following the 10 input states.
- No input sequence crosses a source-day/file boundary.
- Target labels are not included as input features.

### Baseline Model Results

At the default classification threshold of `0.50`:

| Metric | Result |
|---|---:|
| Accuracy | 68.67% |
| Attack Precision | 86.26% |
| Attack Recall | 37.07% |
| Attack F1 | 51.85% |
| Macro-F1 | 64.32% |
| ROC-AUC | 74.47% |
| PR-AUC | 75.92% |
| False Positive Rate | 4.93% |

The default threshold produced a conservative detector with high attack precision but relatively low attack recall.

### Validation-Based Threshold Selection

Because cybersecurity detection generally places significant importance on detecting attacks, the classification threshold was subsequently evaluated on the validation set.

Thresholds from `0.20` through `0.70` were evaluated.

The threshold `0.25` produced the highest validation macro-F1:

- Validation macro-F1: `83.86%`
- Validation attack precision: `84.40%`
- Validation attack recall: `67.15%`

The threshold was then frozen and applied to the untouched test set.

### Test Results at Selected Threshold

At threshold `0.25`:

| Metric | Result |
|---|---:|
| Accuracy | 72.66% |
| Attack Precision | 69.44% |
| Attack Recall | 71.28% |
| Attack F1 | 70.35% |
| Macro-F1 | 72.49% |
| Benign Recall | 73.81% |

This operating point substantially improves attack recall compared with the default `0.50` threshold.

### Current Interpretation

The baseline demonstrates that temporal network-state information contains useful predictive signal for forecasting whether the immediately following network-state window is benign or malicious.

However, the model is not yet considered production-ready.

The main limitations currently identified are:

1. Generalization performance decreases between validation and the later test period.
2. The attack distribution differs between training and later test periods.
3. The current representation uses 21 aggregated features.
4. Option A relies on complete-flow-derived statistics anchored using derived flow availability time.
5. The original CICFlowMeter timestamp retains a documented 12-hour/no-AM-PM ambiguity.
6. The current target is binary rather than a full attack-stage forecast.

### Next ML Experiments

The baseline will be preserved before further experimentation.

Planned experiments include:

- LSTM architecture improvement
- hyperparameter comparison
- threshold/operating-point analysis
- improved attack recall
- comparison against a simpler baseline
- model robustness under temporal distribution shift

The packet-level Option-B representation may subsequently be evaluated as a more rigorous future-information-safe representation.

### Current Artifacts

Training artifacts are stored under:

`ai/outputs/lstm_option_a/`

Important files include:

- `model.pt` — trained baseline LSTM
- `scaler.pkl` — training-fitted feature scaler
- `metrics.json` — final metrics
- `classification_report.txt` — classification report
- `confusion_matrix.png` — test confusion matrix
- `training_curve.png` — training/validation loss
- `training_history.json` — epoch history
- `training_config.json` — experiment configuration
- `threshold_analysis.json` — validation threshold analysis

Raw CSE-CIC-IDS2018 traffic data is not committed to the repository.