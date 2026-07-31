"""Regenerate visualizations for a dataset without phasing ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_phase_visualization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset-name")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_phase_visualization(
        result_dir=args.result_dir,
        dataset_name=args.dataset_name,
    )
    print(
        f"status={result['status']} generated={len(result['generated'])} "
        f"skipped={len(result['skipped'])} failed={len(result['failed'])} "
        f"output_dir={result['output_dir']}"
    )
    for item in result["skipped"]:
        print(f"skipped {item['name']}: {item['reason']}")
    for name, reason in result["failed"].items():
        print(f"failed {name}: {reason}")


if __name__ == "__main__":
    main()
