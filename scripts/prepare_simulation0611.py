from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tv_phase.simulation_adapter import adapt_simulation0611_to_tv_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt data/simulation_0611 into the TV-PHASE dataset layout."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_0611",
        help="Root containing gene_position and position_kegg.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_0611_tv_phase",
        help="Output root for adapted TV-PHASE datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known adapter output files in place if output directories already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    jobs = [
        {
            "name": "gene_position",
            "raw_dir": input_root / "gene_position",
            "output_dir": output_root / "gene_position",
        },
        {
            "name": "position_kegg",
            "raw_dir": input_root / "position_kegg",
            "output_dir": output_root / "position_kegg",
        },
    ]

    print("Preparing simulation_0611 TV-PHASE datasets")
    print(f"  input_root : {input_root.resolve()}")
    print(f"  output_root: {output_root.resolve()}")

    for job in jobs:
        print(f"\n[{job['name']}]")
        print(f"  raw_dir   : {Path(job['raw_dir']).resolve()}")
        print(f"  output_dir: {Path(job['output_dir']).resolve()}")

        paths = adapt_simulation0611_to_tv_phase(
            job["raw_dir"],
            job["output_dir"],
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
