from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tv_phase.simulation_adapter import adapt_simulation0615_to_tv_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt data/simulation_0615 into the TV-PHASE dataset layout."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_0615",
        help="Root containing gene_position_log and gene_position_kegg_corr_log.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "simulation_0615_tv_phase",
        help="Output root for adapted TV-PHASE datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known adapter output files in place if output directories already exist.",
    )
    parser.add_argument(
        "--shared-cell-info-path",
        type=Path,
        default=None,
        help="cell_info.csv used by gene_position_kegg_corr_log. Defaults to gene_position_log/input/cell_info.csv.",
    )
    parser.add_argument(
        "--shared-gene-info-path",
        type=Path,
        default=None,
        help="gene_info.csv used by gene_position_kegg_corr_log. Defaults to gene_position_log/input/gene_info.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    shared_cell_info_path = (
        Path(args.shared_cell_info_path)
        if args.shared_cell_info_path is not None
        else input_root / "gene_position_log" / "input" / "cell_info.csv"
    )
    shared_gene_info_path = (
        Path(args.shared_gene_info_path)
        if args.shared_gene_info_path is not None
        else input_root / "gene_position_log" / "input" / "gene_info.csv"
    )

    jobs = [
        {
            "name": "gene_position",
            "raw_dir": input_root / "gene_position_log",
            "output_dir": output_root / "gene_position",
            "cell_info_path": input_root / "gene_position_log" / "input" / "cell_info.csv",
            "gene_info_path": input_root / "gene_position_log" / "input" / "gene_info.csv",
        },
        {
            "name": "position_kegg",
            "raw_dir": input_root / "gene_position_kegg_corr_log",
            "output_dir": output_root / "position_kegg",
            "cell_info_path": shared_cell_info_path,
            "gene_info_path": shared_gene_info_path,
        },
    ]

    print("Preparing simulation_0615 TV-PHASE datasets")
    print(f"  input_root : {input_root.resolve()}")
    print(f"  output_root: {output_root.resolve()}")

    for job in jobs:
        print(f"\n[{job['name']}]")
        print(f"  raw_dir   : {Path(job['raw_dir']).resolve()}")
        print(f"  output_dir: {Path(job['output_dir']).resolve()}")
        print(f"  cell_info : {Path(job['cell_info_path']).resolve()}")
        print(f"  gene_info : {Path(job['gene_info_path']).resolve()}")

        paths = adapt_simulation0615_to_tv_phase(
            job["raw_dir"],
            job["output_dir"],
            cell_info_path=job["cell_info_path"],
            gene_info_path=job["gene_info_path"],
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
            "simulation0615_source_summary.csv",
        ]:
            print(f"  wrote {key}: {paths[key].resolve()}")
        print(f"  wrote ratio.csv: {(Path(job['output_dir']) / 'ratio.csv').resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
