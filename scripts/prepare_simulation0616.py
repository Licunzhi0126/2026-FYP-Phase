from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tv_phase.simulation_adapter import adapt_simulation0616_to_tv_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adapt all four simulation_0616 cases for TV-PHASE.")
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "data" / "simulation_0616")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "simulation_0616_tv_phase")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known adapter output files in place; directories and unrelated files are not deleted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Preparing simulation_0616 TV-PHASE datasets")
    print(f"  input_root : {args.input_root.resolve()}")
    print(f"  output_root: {args.output_root.resolve()}")
    results = adapt_simulation0616_to_tv_phase(
        args.input_root,
        args.output_root,
        overwrite=args.overwrite,
    )
    for case_name, paths in results.items():
        manifest = json.loads(paths["adapter_manifest.json"].read_text(encoding="utf-8"))
        summary_path = paths["simulation0616_source_summary.csv"]
        shape = manifest["shape"]
        ppi_edges = manifest["ppi_info"]["ppi_edges_written"]
        print(f"\n[{case_name}]")
        print(f"  output      : {paths['expression_data.csv'].parent.resolve()}")
        print(f"  cells/genes : {shape['cells']}/{shape['genes']}")
        print(f"  pathways    : {manifest['n_pathways']}")
        print(f"  PPI edges   : {ppi_edges}")
        print(f"  E_P/E_M/ratio: {paths['E_P.csv'].exists()}/{paths['E_M.csv'].exists()}/{paths['ratio.csv'].exists()}")
        print(f"  summary     : {summary_path.resolve()}")
        print(f"  manifest    : {paths['adapter_manifest.json'].resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
