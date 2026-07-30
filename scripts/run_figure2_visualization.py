"""One-command entry point for both Figure 2 visualization workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_DIR = PROJECT_ROOT / "scripts" / "visualization"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answerdata and simulationdata Figure 2 outputs in separate "
            "directories."
        )
    )
    parser.add_argument(
        "--answer-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "answerdata",
    )
    parser.add_argument(
        "--per-cell-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "per_cell_threshold_0.1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output",
    )
    parser.add_argument("--skip-answerdata", action="store_true")
    parser.add_argument("--skip-simulationdata", action="store_true")

    parser.add_argument("--gse45719-pred-a", type=Path)
    parser.add_argument("--gse45719-pred-b", type=Path)
    parser.add_argument("--gse80810-pred-a", type=Path)
    parser.add_argument("--gse80810-pred-b", type=Path)
    parser.add_argument("--grn-pred-npz", type=Path)
    parser.add_argument("--model-name", default="PhaseHyper")
    parser.add_argument("--answer-primary-method")
    parser.add_argument("--simulation-primary-method")

    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-prevalence", type=float, default=0.95)
    parser.add_argument("--min-allelic-reads", type=int, default=2)
    parser.add_argument("--min-gse80810-reads", type=int, default=8)
    parser.add_argument("--min-scoreable-genes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _append_path_argument(
    command: list[str],
    flag: str,
    value: Path | None,
) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = _parse_args()
    if args.skip_answerdata and args.skip_simulationdata:
        raise ValueError("Both workflows were skipped; there is nothing to run.")

    if not args.skip_answerdata:
        answer_command = [
            sys.executable,
            str(VISUALIZATION_DIR / "figure2_expression_benchmark.py"),
            "--answer-root",
            str(args.answer_root),
            "--output-dir",
            str(args.output_root / "answerdata"),
            "--model-name",
            args.model_name,
            "--n-genes",
            str(args.n_genes),
            "--min-allelic-reads",
            str(args.min_allelic_reads),
            "--min-gse80810-reads",
            str(args.min_gse80810_reads),
            "--min-scoreable-genes",
            str(args.min_scoreable_genes),
            "--seed",
            str(args.seed),
            "--dpi",
            str(args.dpi),
        ]
        _append_path_argument(
            answer_command, "--gse45719-pred-a", args.gse45719_pred_a
        )
        _append_path_argument(
            answer_command, "--gse45719-pred-b", args.gse45719_pred_b
        )
        _append_path_argument(
            answer_command, "--gse80810-pred-a", args.gse80810_pred_a
        )
        _append_path_argument(
            answer_command, "--gse80810-pred-b", args.gse80810_pred_b
        )
        if args.answer_primary_method:
            answer_command.extend(
                ("--primary-method", args.answer_primary_method)
            )
        _run(answer_command)

    if not args.skip_simulationdata:
        simulation_command = [
            sys.executable,
            str(VISUALIZATION_DIR / "figure2_grn_benchmark.py"),
            "--per-cell-root",
            str(args.per_cell_root),
            "--output-dir",
            str(args.output_root / "simulationdata"),
            "--model-name",
            args.model_name,
            "--max-edges",
            str(args.max_edges),
            "--min-prevalence",
            str(args.min_prevalence),
            "--max-prevalence",
            str(args.max_prevalence),
            "--seed",
            str(args.seed),
            "--dpi",
            str(args.dpi),
        ]
        _append_path_argument(
            simulation_command, "--pred-npz", args.grn_pred_npz
        )
        if args.simulation_primary_method:
            simulation_command.extend(
                ("--primary-method", args.simulation_primary_method)
            )
        _run(simulation_command)


if __name__ == "__main__":
    main()
