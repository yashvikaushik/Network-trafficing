"""Run the Option-A, end-anchored CSE-CIC-IDS2018 state pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ai.src.preprocessing.option_a_reader import normalized_records
from ai.src.sequences.option_a_sequences import build_sequences, save_sequences_npz
from ai.src.states.option_a_states import FEATURE_COLUMNS, WindowAccumulator, floor_minute


def run(raw_dir: Path, output_dir: Path, chunk_size: int = 25_000, history: int = 10) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    per_file: list[dict[str, Any]] = []
    state_id = 0

    for path in sorted(raw_dir.glob("*.csv")):
        windows: dict[Any, WindowAccumulator] = {}
        file_counts: Counter[str] = Counter()
        for record, counters in normalized_records(path, chunk_size):
            if not record:
                file_counts.update(counters)
                totals.update(counters)
                continue
            window = floor_minute(record["available_time"])
            windows.setdefault(window, WindowAccumulator()).add(record)
        segment_id = path.stem
        sorted_windows = sorted(windows.items())
        for window_start, accumulator in sorted_windows:
            state_id += 1
            states.append({
                "state_id": state_id,
                "segment_id": segment_id,
                "source_file": path.name,
                "window_start": window_start.isoformat(sep=" "),
                "window_end": window_start.replace(second=59).isoformat(sep=" "),
                "window_label": accumulator.window_label(),
                "flows_in_window": accumulator.flow_count,
                **accumulator.feature_row(),
            })
        per_file.append({
            "source_file": path.name,
            "segment_id": segment_id,
            "rows_read": file_counts["rows_read"],
            "rows_removed": sum(file_counts[key] for key in (
                "repeated_headers", "malformed_rows", "invalid_timestamps", "unexpected_dates",
                "invalid_numeric", "non_finite_numeric")),
            "removal_counts": dict(file_counts),
            "network_states": len(sorted_windows),
            "available_time_range": ([sorted_windows[0][0].isoformat(sep=" "), sorted_windows[-1][0].isoformat(sep=" ")]
                                    if sorted_windows else None),
        })

    labels = sorted({str(state["window_label"]) for state in states})
    label_mapping = {label: index for index, label in enumerate(labels)}
    inputs, targets, target_state_ids = build_sequences(states, label_mapping, history)

    state_path = output_dir / "network_states.csv"
    columns = ["state_id", "segment_id", "source_file", "window_start", "window_end", "window_label", "flows_in_window", *FEATURE_COLUMNS]
    with state_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(states)
    sequence_path = output_dir / "sequences.npz"
    sequence_shape = save_sequences_npz(sequence_path, inputs, targets, target_state_ids, history)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2, sort_keys=True)

    state_label_counts = Counter(str(state["window_label"]) for state in states)
    target_label_counts = Counter(labels[target] for target in targets)
    metadata = {
        "pipeline": "option_a_end_anchored_complete_flow",
        "raw_directory": str(raw_dir),
        "window_size": "1 minute",
        "history_states": history,
        "time_anchor": "Timestamp + Flow Duration (microseconds)",
        "rows_read": totals["rows_read"],
        "rows_removed": sum(totals[key] for key in (
            "repeated_headers", "malformed_rows", "invalid_timestamps", "unexpected_dates",
            "invalid_numeric", "non_finite_numeric")),
        "removal_counts": dict(totals),
        "network_states": len(states),
        "sequences": len(inputs),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": label_mapping,
        "state_label_counts": dict(sorted(state_label_counts.items())),
        "sequence_target_label_counts": dict(sorted(target_label_counts.items())),
        "sequence_shape": {"X": list(sequence_shape), "y": [len(targets)], "target_state_id": [len(target_state_ids)]},
        "per_file": per_file,
        "warnings": [
            "Raw CICFlowMeter timestamps use a 12-hour clock without AM/PM; chronology is only reliable within the displayed clock representation.",
            "States are segmented by source file; sequences never cross files or collection days.",
            "Complete-flow values are anchored to derived flow availability, not the displayed flow-start timestamp.",
            "Rows with non-finite required numeric values are discarded. No imputation is performed.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--history", type=int, default=10)
    args = parser.parse_args()
    metadata = run(args.raw_dir, args.output_dir, args.chunk_size, args.history)
    print(json.dumps({key: metadata[key] for key in ("rows_read", "rows_removed", "network_states", "sequences", "feature_count", "sequence_shape")}, indent=2))


if __name__ == "__main__":
    main()
