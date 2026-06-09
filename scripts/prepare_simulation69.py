from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tv_phase.simulation_adapter import adapt_simulation69_to_tv_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt data/simulation_6.9 into the TV-PHASE dataset layout."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_6.9",
        help="Root containing simulation3 and simulation3_kegg_corr.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_6.9_tv_phase",
        help="Output root for adapted TV-PHASE datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known adapter output files in place if output directories already exist.",
    )
    parser.add_argument(
        "--kegg-missing-stage-label",
        default="Unknown",
        help=(
            "Label to use for simulation3_kegg_corr when cell_info.csv is missing. "
            "This is only suitable for smoke tests and makes ARI/NMI/FMI non-informative."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    jobs = [
        {
            "name": "simulation3",
            "raw_dir": input_root / "simulation3",
            "output_dir": output_root / "simulation3",
            "cell_info_path": input_root / "simulation3" / "cell_info.csv",
            "missing_stage_label": None,
        },
        {
            "name": "simulation3_kegg_corr",
            "raw_dir": input_root / "simulation3_kegg_corr",
            "output_dir": output_root / "simulation3_kegg_corr",
            "cell_info_path": input_root / "simulation3_kegg_corr" / "cell_info.csv",
            "missing_stage_label": args.kegg_missing_stage_label,
        },
    ]

    print("Preparing simulation_6.9 TV-PHASE datasets")
    print(f"  input_root : {input_root.resolve()}")
    print(f"  output_root: {output_root.resolve()}")

    for job in jobs:
        cell_info_path = Path(job["cell_info_path"])
        missing_stage_label = job["missing_stage_label"]
        if cell_info_path.exists():
            stage_note = str(cell_info_path.resolve())
        else:
            stage_note = f"constant label {missing_stage_label!r}"

        print(f"\n[{job['name']}]")
        print(f"  raw_dir   : {Path(job['raw_dir']).resolve()}")
        print(f"  output_dir: {Path(job['output_dir']).resolve()}")
        print(f"  stage     : {stage_note}")
        paths = adapt_simulation69_to_tv_phase(
            job["raw_dir"],
            job["output_dir"],
            cell_info_path=cell_info_path,
            missing_stage_label=missing_stage_label,
            overwrite=args.overwrite,
        )
        for key in [
            "expression_data.csv",
            "cell_stage.csv",
            "kegg_prior.txt",
            "poswin_prior.txt",
            "ppi_prior.csv",
            "E_P.csv",
            "E_M.csv",
        ]:
            print(f"  wrote {key}: {paths[key].resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
