#!/usr/bin/env python3
"""Regenerate the experiment figures from the saved numerical result files.

This entry point uses the same plotting function as ``run_experiment.py`` so
the figures cannot silently drift between the main run and later regeneration.
It never reads or modifies the source CSV datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import load_dependencies, save_diagnostic_plots


BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE / "results",
        help="directory containing validation_results.csv and saved diagnostics",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    required = (
        "validation_results.csv",
        "best_by_model.csv",
        "confusion_matrix_best.csv",
    )
    missing = [name for name in required if not (results_dir / name).is_file()]
    if missing:
        parser.error(
            f"Missing {', '.join(missing)} in {results_dir}; run the full experiment first"
        )

    try:
        load_dependencies()
    except RuntimeError as exc:
        parser.error(str(exc))

    import pandas as pd

    metrics = pd.read_csv(results_dir / "validation_results.csv")
    best_by_model = pd.read_csv(results_dir / "best_by_model.csv")
    confusion = pd.read_csv(results_dir / "confusion_matrix_best.csv", index_col=0)
    status = save_diagnostic_plots(metrics, best_by_model, confusion, results_dir)
    print(f"Generated figures: {status['generated']}")
    if status["failed"]:
        print(f"Figure generation failures: {status['failed']}")
        return 1
    if status["skipped"]:
        print(f"Figures skipped: {status['skipped']}")
    print(f"Plot status saved to {results_dir / 'plot_generation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
