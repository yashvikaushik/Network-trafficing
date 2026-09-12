"""Read CSE-CIC-IDS2018 CSV files without modifying their source data."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
IDENTIFIER_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
REQUIRED_COLUMNS = {
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts",
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Byts/s",
    "Flow Pkts/s", "Flow IAT Mean", "Active Mean", "Idle Mean",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "Label",
}


def expected_file_date(path: Path) -> date:
    """Get the collection date encoded in a dataset filename."""
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", path.name)
    if not match:
        raise ValueError(f"No collection date in filename: {path.name}")
    day, month, year = (int(value) for value in match.groups())
    return date(year, month, day)


def iter_csv_chunks(path: Path, chunk_size: int) -> Iterator[tuple[list[str], list[list[str]]]]:
    """Yield source rows in bounded chunks; headers are returned with every chunk."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [value.strip() for value in next(reader)]
        chunk: list[list[str]] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield header, chunk
                chunk = []
        if chunk:
            yield header, chunk


def normalized_records(path: Path, chunk_size: int) -> Iterator[tuple[dict[str, float | int | str | datetime], dict[str, int]]]:
    """Yield validated, end-anchored records and per-chunk removal counters.

    The source timestamp is parsed as displayed. The dataset's missing AM/PM marker
    cannot be reconstructed here; callers must retain the warning in metadata.
    """
    file_day = expected_file_date(path)
    counters = {"rows_read": 0, "repeated_headers": 0, "malformed_rows": 0,
                "invalid_timestamps": 0, "unexpected_dates": 0,
                "invalid_numeric": 0, "non_finite_numeric": 0}
    for header, rows in iter_csv_chunks(path, chunk_size):
        missing = REQUIRED_COLUMNS.difference(header)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
        index = {name: position for position, name in enumerate(header)}
        for raw_row in rows:
            counters["rows_read"] += 1
            if len(raw_row) != len(header):
                counters["malformed_rows"] += 1
                continue
            row = [value.strip() for value in raw_row]
            label = row[index["Label"]]
            if label == "Label":
                counters["repeated_headers"] += 1
                continue
            try:
                start = datetime.strptime(row[index["Timestamp"]], TIMESTAMP_FORMAT)
            except ValueError:
                counters["invalid_timestamps"] += 1
                continue
            if start.date() != file_day:
                counters["unexpected_dates"] += 1
                continue
            values: dict[str, float] = {}
            numeric_names = REQUIRED_COLUMNS.difference(IDENTIFIER_COLUMNS)
            try:
                for name in numeric_names:
                    value = float(row[index[name]])
                    if not math.isfinite(value):
                        counters["non_finite_numeric"] += 1
                        raise ArithmeticError(name)
                    values[name] = value
            except ArithmeticError:
                continue
            except (ValueError, IndexError):
                counters["invalid_numeric"] += 1
                continue
            duration_us = values["Flow Duration"]
            if duration_us < 0:
                counters["invalid_numeric"] += 1
                continue
            yield ({"start_time": start,
                    "available_time": start + timedelta(microseconds=duration_us),
                    "label": label,
                    **values}, counters)
        yield ({}, counters)
        counters = {key: 0 for key in counters}
