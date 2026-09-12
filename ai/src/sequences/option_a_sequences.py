"""Create fixed-history sequences without crossing source-day boundaries."""

from __future__ import annotations

import io
import struct
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai.src.states.option_a_states import FEATURE_COLUMNS


def build_sequences(states: list[dict[str, Any]], label_mapping: dict[str, int], history: int = 10) -> tuple[list[list[list[float]]], list[int], list[int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["segment_id"])].append(state)
    inputs: list[list[list[float]]] = []
    targets: list[int] = []
    target_state_ids: list[int] = []
    for segment_states in grouped.values():
        ordered = sorted(segment_states, key=lambda state: state["window_start"])
        for target_index in range(history, len(ordered)):
            history_states = ordered[target_index - history:target_index]
            inputs.append([[float(state[name]) for name in FEATURE_COLUMNS] for state in history_states])
            target = ordered[target_index]
            targets.append(label_mapping[str(target["window_label"])])
            target_state_ids.append(int(target["state_id"]))
    return inputs, targets, target_state_ids


def _npy_header(shape: tuple[int, ...], descriptor: str) -> bytes:
    text = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': " + repr(shape)
    text += ", }"
    raw = text.encode("latin1")
    padding = (16 - ((10 + len(raw) + 1) % 16)) % 16
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(raw) + padding + 1) + raw + b" " * padding + b"\n"


def _npy_bytes_float64(values: list[float], shape: tuple[int, ...]) -> bytes:
    return _npy_header(shape, "<f8") + struct.pack("<" + "d" * len(values), *values)


def _npy_bytes_int64(values: list[int], shape: tuple[int, ...]) -> bytes:
    return _npy_header(shape, "<i8") + struct.pack("<" + "q" * len(values), *values)


def save_sequences_npz(path: Path, inputs: list[list[list[float]]], targets: list[int], target_state_ids: list[int], history: int) -> tuple[int, int, int]:
    """Write a NumPy-compatible NPZ using only the standard library."""
    feature_count = len(FEATURE_COLUMNS)
    flattened = [value for sample in inputs for state in sample for value in state]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("X.npy", _npy_bytes_float64(flattened, (len(inputs), history, feature_count)))
        archive.writestr("y.npy", _npy_bytes_int64(targets, (len(targets),)))
        archive.writestr("target_state_id.npy", _npy_bytes_int64(target_state_ids, (len(target_state_ids),)))
    return len(inputs), history, feature_count


def save_flat_sequences_npz(path: Path, inputs: array, targets: array, target_state_ids: array, history: int) -> tuple[int, int, int]:
    """Write already-flattened, NumPy-compatible split arrays without NumPy."""
    feature_count = len(FEATURE_COLUMNS)
    sample_count = len(targets)
    expected_values = sample_count * history * feature_count
    if len(inputs) != expected_values or len(target_state_ids) != sample_count:
        raise ValueError("Split array dimensions are inconsistent")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("X.npy", _npy_header((sample_count, history, feature_count), "<f8") + inputs.tobytes())
        archive.writestr("y.npy", _npy_header((sample_count,), "<i8") + targets.tobytes())
        archive.writestr("target_state_id.npy", _npy_header((sample_count,), "<i8") + target_state_ids.tobytes())
    return sample_count, history, feature_count
