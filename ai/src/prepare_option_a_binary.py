"""CLI entry point for the binary Option-A dataset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.src.sequences.option_a_binary import run_binary_preparation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_binary_preparation(args.source_dir, args.output_dir)
    print(json.dumps({name: value["samples"] for name, value in result["split_metadata"]["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
