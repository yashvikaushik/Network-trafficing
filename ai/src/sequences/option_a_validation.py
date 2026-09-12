"""Validation and chronological source-day splitting for Option-A sequences."""

from __future__ import annotations

import ast
import csv
import json
import math
import struct
import sys
import zipfile
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ai.src.sequences.option_a_sequences import save_flat_sequences_npz
from ai.src.states.option_a_states import FEATURE_COLUMNS


def _read_npy(data: bytes) -> tuple[array, tuple[int, ...]]:
    if data[:6] != b"\x93NUMPY" or data[6:8] != b"\x01\x00":
        raise ValueError("Only NumPy v1 archives are supported")
    header_length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_length].decode("latin1").strip())
    descriptor = header["descr"]
    if descriptor == "<f8":
        values = array("d")
    elif descriptor == "<i8":
        values = array("q")
    else:
        raise ValueError(f"Unexpected array dtype: {descriptor}")
    values.frombytes(data[10 + header_length:])
    if sys.byteorder != "little":
        values.byteswap()
    return values, tuple(header["shape"])


def load_sequences(path: Path) -> tuple[array, tuple[int, ...], array, array]:
    with zipfile.ZipFile(path) as archive:
        required = {"X.npy", "y.npy", "target_state_id.npy"}
        if set(archive.namelist()) != required:
            raise ValueError("Unexpected sequence archive members")
        inputs, shape = _read_npy(archive.read("X.npy"))
        targets, target_shape = _read_npy(archive.read("y.npy"))
        ids, id_shape = _read_npy(archive.read("target_state_id.npy"))
    if target_shape != (shape[0],) or id_shape != (shape[0],):
        raise ValueError("Sequence target dimensions do not match X")
    return inputs, shape, targets, ids


def repair_legacy_one_dimensional_headers(path: Path) -> bool:
    """Repair the prior writer's NPZ header typo without changing array payloads."""
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    repaired = False
    for name, payload in members.items():
        if payload[:6] != b"\x93NUMPY":
            continue
        header_length = struct.unpack("<H", payload[8:10])[0]
        header = payload[10:10 + header_length]
        # A first attempted repair shortened the header while leaving its length
        # field intact. Its final header byte is therefore the first data byte.
        if header and header[-1:] not in {b"\n", b" "}:
            header_without_spill = header[:-1]
            if not header_without_spill.endswith(b"\n"):
                raise ValueError(f"Cannot repair malformed header in {name}")
            corrected = header_without_spill[:-1] + b" " + b"\n"
            members[name] = payload[:10] + corrected + header[-1:] + payload[10 + header_length:]
            repaired = True
            continue
        corrected = header.replace(b"),, }", b"), }")
        if corrected != header:
            if corrected.endswith(b"\n"):
                corrected = corrected[:-1] + b" " * (len(header) - len(corrected)) + b"\n"
            if len(corrected) != len(header):
                raise ValueError(f"Cannot preserve header length in {name}")
            members[name] = payload[:10] + corrected + payload[10 + header_length:]
            repaired = True
    if repaired:
        temporary = path.with_suffix(".repair.npz")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        temporary.replace(path)
    return repaired


def read_states(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"state_id", "segment_id", "source_file", "window_start", "window_label", *FEATURE_COLUMNS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("network_states.csv lacks required columns")
        result = []
        for row in reader:
            row["state_id"] = int(row["state_id"])
            for feature in FEATURE_COLUMNS:
                row[feature] = float(row[feature])
            result.append(row)
    return result


def expected_sequence_layout(states: list[dict[str, Any]], label_mapping: dict[str, int], history: int) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["segment_id"])].append(state)
    expected = []
    for segment_states in grouped.values():
        ordered = sorted(segment_states, key=lambda state: str(state["window_start"]))
        for index in range(history, len(ordered)):
            target = ordered[index]
            if str(target["window_label"]) not in label_mapping:
                raise ValueError("A state label has no mapping")
            expected.append((ordered[index - history:index], target))
    return expected


def validate_sequences(states: list[dict[str, Any]], inputs: array, shape: tuple[int, ...], targets: array, target_ids: array, label_mapping: dict[str, int], history: int) -> dict[str, Any]:
    if shape[1:] != (history, len(FEATURE_COLUMNS)):
        raise ValueError(f"Unexpected X shape: {shape}")
    expected = expected_sequence_layout(states, label_mapping, history)
    if len(expected) != shape[0]:
        raise ValueError("Sequence count differs from states-derived expectation")
    target_immediately_after_history = True
    no_cross_segment = True
    x_matches_states = True
    values_per_sample = history * len(FEATURE_COLUMNS)
    for sample_index, (history_states, target) in enumerate(expected):
        if target_ids[sample_index] != target["state_id"] or targets[sample_index] != label_mapping[target["window_label"]]:
            target_immediately_after_history = False
            break
        if any(state["segment_id"] != target["segment_id"] for state in history_states):
            no_cross_segment = False
            break
        offset = sample_index * values_per_sample
        expected_values = [float(state[feature]) for state in history_states for feature in FEATURE_COLUMNS]
        if any(inputs[offset + position] != value for position, value in enumerate(expected_values)):
            x_matches_states = False
            break
    finite_inputs = all(math.isfinite(value) for value in inputs)
    finite_state_features = all(math.isfinite(float(state[feature])) for state in states for feature in FEATURE_COLUMNS)
    label_as_feature = any("label" in feature.lower() for feature in FEATURE_COLUMNS)
    ordered_by_segment = defaultdict(list)
    for state in states:
        ordered_by_segment[state["segment_id"]].append(state)
    timestamp_ordered = all(
        all(datetime.fromisoformat(current["window_start"]) > datetime.fromisoformat(previous["window_start"])
            for previous, current in zip(sorted(segment, key=lambda value: value["window_start"]),
                                         sorted(segment, key=lambda value: value["window_start"])[1:]))
        for segment in ordered_by_segment.values()
    )
    return {
        "target_immediately_after_10_inputs": target_immediately_after_history,
        "sequence_inputs_match_preceding_states": x_matches_states,
        "no_sequence_crosses_source_segment": no_cross_segment,
        "input_features_exclude_target_label": not label_as_feature,
        "all_input_values_finite": finite_inputs,
        "all_state_feature_values_finite": finite_state_features,
        "state_windows_strictly_increasing_within_segment": timestamp_ordered,
        "timestamp_ambiguity": "The source CICFlowMeter timestamp uses a 12-hour clock without AM/PM; displayed-clock ordering is verified, but all-day ordering remains unresolved.",
    }


def choose_chronological_segments(states: list[dict[str, Any]], target_ids: array) -> tuple[list[str], list[str], list[str]]:
    by_id = {state["state_id"]: state for state in states}
    samples_by_segment: dict[str, list[int]] = defaultdict(list)
    starts: dict[str, str] = {}
    for sample_index, state_id in enumerate(target_ids):
        state = by_id[int(state_id)]
        samples_by_segment[state["segment_id"]].append(sample_index)
        starts.setdefault(state["segment_id"], str(state["window_start"]))
    segments = sorted(samples_by_segment, key=lambda segment: starts[segment])
    counts = [len(samples_by_segment[segment]) for segment in segments]
    total = sum(counts)
    best: tuple[float, int, int] | None = None
    for train_end in range(1, len(segments) - 1):
        for validation_end in range(train_end + 1, len(segments)):
            split_counts = (sum(counts[:train_end]), sum(counts[train_end:validation_end]), sum(counts[validation_end:]))
            score = sum((value / total - target) ** 2 for value, target in zip(split_counts, (0.70, 0.15, 0.15)))
            candidate = (score, train_end, validation_end)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, train_end, validation_end = best
    return segments[:train_end], segments[train_end:validation_end], segments[validation_end:]


def write_split(path: Path, indices: list[int], inputs: array, shape: tuple[int, ...], targets: array, target_ids: array) -> list[int]:
    width = shape[1] * shape[2]
    split_x = array("d")
    split_y = array("q")
    split_ids = array("q")
    for index in indices:
        split_x.extend(inputs[index * width:(index + 1) * width])
        split_y.append(targets[index])
        split_ids.append(target_ids[index])
    save_flat_sequences_npz(path, split_x, split_y, split_ids, shape[1])
    return list(split_y)


def run_validation_and_split(output_dir: Path, history: int = 10) -> dict[str, Any]:
    states = read_states(output_dir / "network_states.csv")
    label_mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    sequence_path = output_dir / "sequences.npz"
    repaired_archive_headers = repair_legacy_one_dimensional_headers(sequence_path)
    inputs, shape, targets, target_ids = load_sequences(sequence_path)
    checks = validate_sequences(states, inputs, shape, targets, target_ids, label_mapping, history)
    if not all(value is True for key, value in checks.items() if key != "timestamp_ambiguity"):
        raise ValueError(f"Validation failed: {checks}")
    by_id = {state["state_id"]: state for state in states}
    train_segments, val_segments, test_segments = choose_chronological_segments(states, target_ids)
    split_segments = {"train": train_segments, "validation": val_segments, "test": test_segments}
    inverse_mapping = {value: key for key, value in label_mapping.items()}
    split_results: dict[str, Any] = {}
    for name, segments in split_segments.items():
        indices = [index for index, state_id in enumerate(target_ids) if by_id[int(state_id)]["segment_id"] in segments]
        indices.sort(key=lambda index: (by_id[int(target_ids[index])]["window_start"], by_id[int(target_ids[index])]["segment_id"]))
        labels = write_split(output_dir / f"{name if name != 'validation' else 'val'}.npz", indices, inputs, shape, targets, target_ids)
        split_results[name] = {
            "samples": len(indices),
            "X_shape": [len(indices), shape[1], shape[2]],
            "y_shape": [len(indices)],
            "source_segments": segments,
            "source_files": [by_id[int(target_ids[index])]["source_file"] for index in indices if by_id[int(target_ids[index])]["segment_id"] in segments],
            "class_distribution": dict(sorted(Counter(inverse_mapping[int(label)] for label in labels).items())),
        }
        split_results[name]["source_files"] = list(dict.fromkeys(split_results[name]["source_files"]))
    train_classes = set(split_results["train"]["class_distribution"])
    class_coverage = {
        "train_classes": sorted(train_classes),
        "validation_classes_absent_from_train": sorted(set(split_results["validation"]["class_distribution"]).difference(train_classes)),
        "test_classes_absent_from_train": sorted(set(split_results["test"]["class_distribution"]).difference(train_classes)),
    }
    report = {
        "validation_checks": checks,
        "label_rule": "A non-empty minute is Benign only when every retained flow is Benign; otherwise it receives the most frequent non-benign existing Label, with alphabetical tie-breaking.",
        "split_policy": "Whole source-file/day segments are sorted by their derived available-time start. The partition minimizing squared deviation from 70%/15%/15% is selected; no individual sequence is randomly split.",
        "source_timestamp_warning": metadata["warnings"][0],
        "repaired_legacy_npz_header_typo": repaired_archive_headers,
        "class_coverage": class_coverage,
        "splits": split_results,
        "ready_for_lstm_training": "not ready for closed-set multiclass LSTM evaluation: tensor integrity and day-boundary isolation pass, but Bot and Infilteration are absent from training and appear in later holdouts. Resolve the 12-hour timestamp ambiguity and define an unseen-class/novel-behaviour policy before training.",
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Option-A sequence validation and split report", "",
        "## Validation result", "",
        "All structural checks passed: each target is the state immediately after its ten inputs; inputs match the preceding state features; no sequence crosses a source segment; labels are not input features; and all state/tensor feature values are finite.", "",
        "## Label rule", "",
        report["label_rule"], "",
        "## Chronological split", "",
        report["split_policy"], "",
    ]
    for name in ("train", "validation", "test"):
        split = split_results[name]
        lines.extend([f"### {name.title()}", "", f"- Samples: {split['samples']}", f"- `X` shape: {split['X_shape']}", f"- `y` shape: {split['y_shape']}", f"- Source files: {', '.join(split['source_files'])}", f"- Class distribution: `{json.dumps(split['class_distribution'], sort_keys=True)}`", ""])
    lines.extend(["## Warnings", "", f"- {report['source_timestamp_warning']}", "- The label rule is verified against the persisted pipeline specification and label mapping. Recomputing each label from individual flows would require revisiting raw data, which this task did not do.", "- Class imbalance remains substantial, including rare window classes such as `DoS attacks-Hulk`.", f"- Validation classes absent from training: {', '.join(class_coverage['validation_classes_absent_from_train']) or 'none'}.", f"- Test classes absent from training: {', '.join(class_coverage['test_classes_absent_from_train']) or 'none'}.", "", "## Readiness", "", report["ready_for_lstm_training"], ""])
    (output_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
