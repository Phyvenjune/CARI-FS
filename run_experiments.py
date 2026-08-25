from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import run_cari_fs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARI-FS experiments with a leakage-free nested protocol.")
    parser.add_argument("--data-dir", default="../dataset", help="Directory containing CSV datasets")
    parser.add_argument("--output", default="cari_fs_results.json", help="Output JSON path")
    parser.add_argument("--dataset", default="all", help="CSV stem to run, or 'all'")
    parser.add_argument("--target", default=None, help="Optional target column; default is the last column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if args.dataset != "all":
        files = [data_dir / f"{args.dataset.removesuffix('.csv')}.csv"]
    results: dict[str, dict] = {}
    for path in files:
        if not path.exists():
            results[path.stem] = {"error": f"Dataset not found: {path}"}
            continue
        try:
            frame = pd.read_csv(path)
            target = args.target or frame.columns[-1]
            results[path.stem] = run_cari_fs(frame, target)
        except Exception as error:
            results[path.stem] = {"error": str(error)}
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
