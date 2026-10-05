#!/usr/bin/env python3
"""Read-only exploratory summary for Project 1.

The script never modifies either CSV.  By default it prints a concise summary;
``--output`` may be used to save the same information as JSON for the report.
Paths are resolved relative to this script, so it can be run from any working
directory.
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def _quantiles(series: Any) -> dict[str, float]:
    if series.empty:
        return {}
    values = series.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {str(float(index)): float(value) for index, value in values.items()}


def build_summary(data_dir: Path) -> dict[str, Any]:
    try:
        metadata.version("pandas")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing required package: pandas. Activate the project Conda environment "
            "and install its locked dependencies before exploring the data."
        ) from exc
    import pandas as pd

    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data_unlabeled.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected train_data.csv and test_data_unlabeled.csv in {data_dir}"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    required_train = {"text", "target"}
    required_test = {"text"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train_data.csv must contain {sorted(required_train)}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"test_data_unlabeled.csv must contain {sorted(required_test)}")

    train_missing_text = train["text"].isna()
    test_missing_text = test["text"].isna()
    train_text = train["text"].fillna("").astype(str)
    test_text = test["text"].fillna("").astype(str)
    train_chars = train_text.str.len()
    test_chars = test_text.str.len()
    train_words = train_text.str.split().str.len()
    test_words = test_text.str.split().str.len()
    numeric_targets = pd.to_numeric(train["target"], errors="coerce")
    integral_targets = numeric_targets.notna() & (numeric_targets % 1 == 0)
    target_values = sorted(train["target"].dropna().unique().tolist(), key=str)
    target_count_items = train["target"].value_counts().items()
    observed_integer_targets = sorted(
        {int(value) for value in numeric_targets[integral_targets].tolist()}
    )

    validation_errors: list[str] = []
    if len(train) == 0:
        validation_errors.append("training data has no rows")
    if len(test) == 0:
        validation_errors.append("test data has no rows")
    if train["target"].isna().any():
        validation_errors.append("training labels contain missing values")
    if (~integral_targets).any():
        validation_errors.append("training labels must be integer IDs")
    if observed_integer_targets != list(range(10)):
        validation_errors.append("training labels must contain every ID from 0 through 9")
    if train_missing_text.any() or (train_text.str.strip() == "").any():
        validation_errors.append("training text contains missing or empty rows")
    if test_missing_text.any() or (test_text.str.strip() == "").any():
        validation_errors.append("test text contains missing or empty rows")

    # This is a diagnostic overlap check only.  It does not remove or alter rows.
    train_unique = set(train_text)
    test_unique = set(test_text)
    summary: dict[str, Any] = {
        "files": {
            "train": str(train_path),
            "test": str(test_path),
        },
        "train": {
            "rows": int(len(train)),
            "columns": list(train.columns),
            "dtypes": {key: str(value) for key, value in train.dtypes.items()},
            "missing_values": {key: int(value) for key, value in train.isna().sum().items()},
            "duplicate_text_rows": int(train_text.duplicated().sum()),
            "empty_text_rows": int((train_chars == 0).sum()),
            "target_values": target_values,
            "valid_target_ids": observed_integer_targets,
            "target_counts": {
                str(key): int(value)
                for key, value in sorted(target_count_items, key=lambda item: str(item[0]))
            },
            "character_length_quantiles": _quantiles(train_chars),
            "word_length_quantiles": _quantiles(train_words),
        },
        "test": {
            "rows": int(len(test)),
            "columns": list(test.columns),
            "dtypes": {key: str(value) for key, value in test.dtypes.items()},
            "missing_values": {key: int(value) for key, value in test.isna().sum().items()},
            "duplicate_text_rows": int(test_text.duplicated().sum()),
            "empty_text_rows": int((test_chars == 0).sum()),
            "character_length_quantiles": _quantiles(test_chars),
            "word_length_quantiles": _quantiles(test_words),
        },
        "cross_split_checks": {
            "train_test_exact_text_overlap": int(len(train_unique & test_unique)),
        },
        "validation": {
            "valid": not validation_errors,
            "expected_target_ids": list(range(10)),
            "errors": validation_errors,
            "training_missing_or_empty_text_rows": int(
                (train_missing_text | (train_text.str.strip() == "")).sum()
            ),
            "test_missing_or_empty_text_rows": int(
                (test_missing_text | (test_text.str.strip() == "")).sum()
            ),
        },
        "notes": [
            "The script reports diagnostics only; it does not clean, deduplicate, or rewrite either CSV.",
            "Training requires integer target IDs 0 through 9 and non-empty text rows.",
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR,
        help="directory containing train_data.csv and test_data_unlabeled.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path; parent directories are created if needed",
    )
    args = parser.parse_args()
    try:
        summary = build_summary(args.data_dir.resolve())
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote read-only summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()
