"""One-command entry point for both Figure 2 visualization workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_DIR = PROJECT_ROOT / "scripts" / "visualization"
SERVER_DATA_ROOT = Path("/home/jovyan/public/datasets/PHASE")


def _default_data_root() -> Path:
    configured = os.environ.get("PHASE_DATA_ROOT")
    if configured:
        return Path(configured)
    if SERVER_DATA_ROOT.exists() or os.name != "nt":
        return SERVER_DATA_ROOT
    return PROJECT_ROOT / "data" / "answerdata"


DEFAULT_DATA_ROOT = _default_data_root()


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
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--per-cell-root",
        type=Path,
        default=(
            DEFAULT_DATA_ROOT / "per_cell_threshold_0.1"
            if DEFAULT_DATA_ROOT != PROJECT_ROOT / "data" / "answerdata"
            else PROJECT_ROOT / "data" / "per_cell_threshold_0.1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output",
    )
    parser.add_argument("--skip-answerdata", action="store_true")
    parser.add_argument("--skip-simulationdata", action="store_true")
    parser.add_argument("--skip-model-fit", action="store_true")
    parser.add_argument("--force-model", action="store_true")

    parser.add_argument("--gse45719-pred-a", type=Path)
    parser.add_argument("--gse45719-pred-b", type=Path)
    parser.add_argument("--gse80810-pred-a", type=Path)
    parser.add_argument("--gse80810-pred-b", type=Path)
    parser.add_argument("--grn-pred-npz", type=Path)
    parser.add_argument("--model-name", default="HyperPhase")
    parser.add_argument("--answer-primary-method")
    parser.add_argument("--simulation-primary-method")

    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-prevalence", type=float, default=0.95)
    parser.add_argument("--min-allelic-reads", type=int, default=2)
    parser.add_argument("--min-gse80810-reads", type=int, default=8)
    parser.add_argument("--min-scoreable-genes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--grn-epochs", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
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
    print("Running:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _validate_inputs(args: argparse.Namespace) -> None:
    missing: list[Path] = []
    if not args.skip_answerdata:
        missing.extend(
            path
            for path in (
                args.answer_root / "GSE45719",
                args.answer_root / "GSE80810",
            )
            if not path.is_dir()
        )
    if not args.skip_simulationdata:
        missing.extend(
            path
            for path in (
                args.per_cell_root / "combined",
                args.per_cell_root / "maternal",
                args.per_cell_root / "paternal",
            )
            if not path.is_dir()
        )
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Figure 2 input directories were not found:\n"
            f"{formatted}\n"
            "Override them with --answer-root/--per-cell-root or set "
            "PHASE_DATA_ROOT."
        )


def main() -> None:
    args = _parse_args()
    if args.skip_answerdata and args.skip_simulationdata:
        raise ValueError("Both workflows were skipped; there is nothing to run.")
    _validate_inputs(args)
    print(f"Answer-data root: {args.answer_root}", flush=True)
    print(f"Simulation-data root: {args.per_cell_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)

    answer_model_root = (
        args.output_root / "answerdata" / "hyperphase_outputs"
    )
    simulation_model_root = (
        args.output_root / "simulationdata" / "hyperphase_outputs"
    )

    if not args.skip_model_fit:
        model_command = [
            sys.executable,
            str(VISUALIZATION_DIR / "hyperphase_adapter.py"),
            "--answer-root",
            str(args.answer_root),
            "--per-cell-root",
            str(args.per_cell_root),
            "--output-root",
            str(args.output_root),
            "--n-genes",
            str(args.n_genes),
            "--grn-max-edges",
            str(args.max_edges),
            "--min-prevalence",
            str(args.min_prevalence),
            "--max-prevalence",
            str(args.max_prevalence),
            "--min-allelic-reads",
            str(args.min_allelic_reads),
            "--min-gse80810-reads",
            str(args.min_gse80810_reads),
            "--min-scoreable-genes",
            str(args.min_scoreable_genes),
            "--epochs",
            str(args.epochs),
            "--grn-epochs",
            str(args.grn_epochs),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.skip_answerdata:
            model_command.append("--skip-answerdata")
        if args.skip_simulationdata:
            model_command.append("--skip-simulationdata")
        if args.force_model:
            model_command.append("--force")
        _run(model_command)

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
        explicit_expression_predictions = any(
            value is not None
            for value in (
                args.gse45719_pred_a,
                args.gse45719_pred_b,
                args.gse80810_pred_a,
                args.gse80810_pred_b,
            )
        )
        if explicit_expression_predictions:
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
        else:
            answer_command.extend(
                ("--hyperphase-root", str(answer_model_root))
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
        simulation_prediction = (
            args.grn_pred_npz
            if args.grn_pred_npz is not None
            else simulation_model_root / "hyperphase_grn_predictions.npz"
        )
        _append_path_argument(
            simulation_command, "--pred-npz", simulation_prediction
        )
        if args.simulation_primary_method:
            simulation_command.extend(
                ("--primary-method", args.simulation_primary_method)
            )
        _run(simulation_command)


if __name__ == "__main__":
    main()
