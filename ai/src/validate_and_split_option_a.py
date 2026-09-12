"""CLI entry point for Option-A validation and chronological splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.src.sequences.option_a_validation import run_validation_and_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_validation_and_split(args.output_dir)
    print(json.dumps({name: details["samples"] for name, details in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
