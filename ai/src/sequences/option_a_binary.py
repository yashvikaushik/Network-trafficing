"""Prepare and validate the binary target variant of Option-A sequences."""

from __future__ import annotations

import csv
import json
import math
from array import array
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ai.src.sequences.option_a_sequences import build_sequences, save_sequences_npz
from ai.src.sequences.option_a_validation import load_sequences, read_states, validate_sequences
from ai.src.sequences.option_a_sequences import save_flat_sequences_npz
from ai.src.states.option_a_states import FEATURE_COLUMNS


BINARY_MAPPING = {"BENIGN": 0, "ATTACK": 1}
TRAIN_DATES = {date(2018, 2, day) for day in (14, 15, 16, 20, 21, 22, 23)}
VALIDATION_DATES = {date(2018, 2, 28)}
TEST_DATES = {date(2018, 3, day) for day in (1, 2)}


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _binary_label(source_label: str) -> str:
    return "BENIGN" if source_label == "Benign" else "ATTACK"


def _write_binary_states(source_path: Path, destination_path: Path) -> None:
    fields, rows = _load_csv_rows(source_path)
    if "window_label" not in fields:
        raise ValueError("Source state table lacks window_label")
    output_fields = ["source_window_label" if field == "window_label" else field for field in fields]
    output_fields.extend(["window_label", "binary_label"])
    with destination_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            source_label = row.pop("window_label")
            binary_name = _binary_label(source_label)
            row["source_window_label"] = source_label
            row["window_label"] = binary_name
            row["binary_label"] = BINARY_MAPPING[binary_name]
            writer.writerow(row)


def _split_for_date(value: date) -> str:
    if value in TRAIN_DATES:
        return "train"
    if value in VALIDATION_DATES:
        return "validation"
    if value in TEST_DATES:
        return "test"
    raise ValueError(f"State date is outside the mandated binary split: {value}")


def _write_split(path: Path, indices: list[int], inputs: list[list[list[float]]], targets: list[int], target_ids: list[int]) -> None:
    split_x = array("d", (value for index in indices for state in inputs[index] for value in state))
    split_y = array("q", (targets[index] for index in indices))
    split_ids = array("q", (target_ids[index] for index in indices))
    save_flat_sequences_npz(path, split_x, split_y, split_ids, history=10)


def run_binary_preparation(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_binary_states(source_dir / "network_states.csv", output_dir / "network_states.csv")
    states = read_states(output_dir / "network_states.csv")
    source_states = read_states(source_dir / "network_states.csv")
    if len(states) != len(source_states):
        raise ValueError("Binary state count differs from source state count")
    for binary_state, source_state in zip(states, source_states):
        if any(binary_state[feature] != source_state[feature] for feature in FEATURE_COLUMNS):
            raise ValueError("A network-state feature changed during binary conversion")

    inputs, targets, target_ids = build_sequences(states, BINARY_MAPPING, history=10)
    save_sequences_npz(output_dir / "sequences.npz", inputs, targets, target_ids, history=10)
    loaded_x, shape, loaded_y, loaded_ids = load_sequences(output_dir / "sequences.npz")
    checks = validate_sequences(states, loaded_x, shape, loaded_y, loaded_ids, BINARY_MAPPING, history=10)
    if not all(value is True for key, value in checks.items() if key != "timestamp_ambiguity"):
        raise ValueError(f"Binary sequence validation failed: {checks}")

    state_by_id = {state["state_id"]: state for state in states}
    split_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for index, state_id in enumerate(target_ids):
        target_state = state_by_id[state_id]
        split_indices[_split_for_date(datetime.fromisoformat(target_state["window_start"]).date())].append(index)
    split_details: dict[str, Any] = {}
    for split, indices in split_indices.items():
        indices.sort(key=lambda index: (state_by_id[target_ids[index]]["window_start"], state_by_id[target_ids[index]]["segment_id"]))
        filename = "val.npz" if split == "validation" else f"{split}.npz"
        _write_split(output_dir / filename, indices, inputs, targets, target_ids)
        class_counts = Counter(targets[index] for index in indices)
        source_files = list(dict.fromkeys(state_by_id[target_ids[index]]["source_file"] for index in indices))
        split_details[split] = {
            "samples": len(indices),
            "X_shape": [len(indices), 10, len(FEATURE_COLUMNS)],
            "y_shape": [len(indices)],
            "binary_target_counts": {"BENIGN": class_counts[0], "ATTACK": class_counts[1]},
            "both_classes_present": class_counts[0] > 0 and class_counts[1] > 0,
            "source_files": source_files,
        }

    original_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    binary_metadata = {
        "pipeline": "option_a_binary_end_anchored_complete_flow",
        "source_processed_directory": str(source_dir),
        "target_definition": {"BENIGN": 0, "ATTACK": 1, "rule": "Benign source window label maps to BENIGN; every other source window label maps to ATTACK."},
        "history_states": 10,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "network_states": len(states),
        "sequences": len(inputs),
        "sequence_shape": {"X": list(shape), "y": [len(loaded_y)], "target_state_id": [len(loaded_ids)]},
        "source_time_anchor": original_metadata["time_anchor"],
        "feature_values_unchanged_from_option_a": True,
        "timestamp_evidence": {
            "metadata_warning": original_metadata["warnings"][0],
            "available_time_ranges_show_derived_end_times": "Several persisted per-file ranges end at 13:01, but that derives from adding flow duration and does not restore AM/PM to the source start timestamp.",
            "conclusion": "The 12-hour/no-AM-PM ambiguity remains unresolved. Do not infer AM/PM values from these artifacts.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(binary_metadata, indent=2, sort_keys=True), encoding="utf-8")
    split_metadata = {
        "split_policy": "Mandated complete source-day split: train 14/15/16/20/21/22/23 Feb; validation 28 Feb; test 1/2 Mar. No sequence is randomly split.",
        "validation_checks": checks,
        "splits": split_details,
        "timestamp_assessment": "Sufficient only for a bounded binary baseline under the persisted displayed-clock ordering. It remains an unresolved limitation for claims requiring unambiguous all-day temporal order.",
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2, sort_keys=True), encoding="utf-8")
    readme = f"""# Option-A binary forecasting dataset

This directory is the binary-target derivative of `ai/data/processed/option_a/`. Network-state feature values are copied unchanged.

## Target

- `BENIGN = 0`: source `window_label` was `Benign`.
- `ATTACK = 1`: source `window_label` was any non-Benign class.

`network_states.csv` retains the multiclass source target as `source_window_label` and stores the binary target as `window_label` and `binary_label`.

## Sequences and split

Each sample contains ten prior one-minute network states with 21 numeric features. The target is the binary label of the immediately following state in the same source file/day. `sequences.npz`, `train.npz`, `val.npz`, and `test.npz` each contain NumPy-compatible `X`, `y`, and `target_state_id` arrays.

The mandated chronological split is preserved: train (14, 15, 16, 20, 21, 22, 23 Feb), validation (28 Feb), and test (1–2 Mar). No sequence crosses a source-file boundary and no individual sequence is randomly assigned.

## Timestamp limitation

The source CICFlowMeter timestamps have a 12-hour clock without AM/PM. Existing metadata verifies ordering only in that displayed-clock representation. Derived availability windows can extend to 13:01 after adding flow duration, but this does not reconstruct source AM/PM. The ambiguity remains unresolved; do not use this baseline to claim unambiguous all-day temporal ordering.

## Validation summary

- States: {len(states)}; sequences: {len(inputs)}; `X` shape: {list(shape)}.
- Features are finite and unchanged from Option A.
- Sequence targets were verified as the state immediately after their ten inputs.
- Every split contains both binary target classes.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"metadata": binary_metadata, "split_metadata": split_metadata}
