#!/usr/bin/env python3
"""Compare the independent Conda runs for Project 1 reproducibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOLERANCE = 1e-10
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
METRIC_COLUMNS = ("accuracy", "macro_f1", "weighted_f1")
TIMING_COLUMNS = {
    "vectorizer_fit_seconds",
    "validation_transform_seconds",
    "estimator_fit_seconds",
    "prediction_seconds",
    "total_seconds",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_difference(left: Any, right: Any) -> float:
    return abs(float(left) - float(right))


def compare_json(left: Any, right: Any, path: str, issues: list[str]) -> float:
    """Compare JSON-compatible summary values, tolerating floating point noise."""

    max_difference = 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            issues.append(f"{path}: JSON object keys differ")
            return max_difference
        for key in left:
            max_difference = max(
                max_difference,
                compare_json(left[key], right[key], f"{path}.{key}", issues),
            )
        return max_difference
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            issues.append(f"{path}: JSON list lengths differ")
            return max_difference
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            max_difference = max(
                max_difference,
                compare_json(left_item, right_item, f"{path}[{index}]", issues),
            )
        return max_difference
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        difference = abs(float(left) - float(right))
        if difference > TOLERANCE:
            issues.append(f"{path}: difference {difference:.17g} exceeds {TOLERANCE:g}")
        return difference
    if left != right:
        issues.append(f"{path}: values differ")
    return max_difference


def rows_by_key(
    rows: list[dict[str, str]], key: str, label: str, issues: list[str]
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in result:
            issues.append(f"{label}: missing or duplicate {key} value {value!r}")
        result[value] = row
    return result


def compare_metric_rows(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    key: str,
    label: str,
    issues: list[str],
    *,
    check_class_f1: bool = False,
) -> tuple[float, int]:
    left = rows_by_key(left_rows, key, f"{label} primary", issues)
    right = rows_by_key(right_rows, key, f"{label} reproduction", issues)
    if set(left) != set(right):
        issues.append(f"{label}: {key} values differ between runs")

    max_difference = 0.0
    compared_values = 0
    stable_columns = {
        "model",
        "vectorizer",
        "max_features",
        "ngram_range",
        "min_df",
        "sublinear_tf",
        "alpha",
        "C",
        "num_features",
        "train_rows",
        "validation_rows",
        "status",
        "convergence_warning",
        "warnings",
    }

    for row_key in sorted(set(left) & set(right)):
        left_row, right_row = left[row_key], right[row_key]
        for column in stable_columns:
            if column in left_row or column in right_row:
                if left_row.get(column) != right_row.get(column):
                    issues.append(f"{label} {row_key}: {column} differs")
        for column in METRIC_COLUMNS:
            try:
                difference = numeric_difference(left_row[column], right_row[column])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{label} {row_key}: invalid or missing {column}")
                continue
            max_difference = max(max_difference, difference)
            compared_values += 1
            if difference > TOLERANCE:
                issues.append(
                    f"{label} {row_key}: {column} differs by {difference:.17g}"
                )

        if check_class_f1:
            try:
                left_f1 = json.loads(left_row["class_f1"])
                right_f1 = json.loads(right_row["class_f1"])
            except (KeyError, json.JSONDecodeError) as exc:
                issues.append(f"{label} {row_key}: cannot parse class_f1 ({exc})")
                continue
            if set(left_f1) != set(right_f1):
                issues.append(f"{label} {row_key}: class_f1 class IDs differ")
                continue
            for class_id in sorted(left_f1):
                difference = abs(float(left_f1[class_id]) - float(right_f1[class_id]))
                max_difference = max(max_difference, difference)
                compared_values += 1
                if difference > TOLERANCE:
                    issues.append(
                        f"{label} {row_key}: class {class_id} F1 differs by "
                        f"{difference:.17g}"
                    )

    return max_difference, compared_values


def compare_rows_exact(
    left: list[dict[str, str]], right: list[dict[str, str]], label: str, issues: list[str]
) -> bool:
    same = left == right
    if not same:
        issues.append(f"{label}: rows differ")
    return same


def build_report(root: Path) -> dict[str, Any]:
    issues: list[str] = []
    primary_dir = root / "results"
    reproduction_dir = root / "results_reproduced"

    required_paths = [
        root / "run_experiment.py",
        root / "explore_data.py",
        root / "train_data.csv",
        root / "test_data_unlabeled.csv",
        root / "environment.yml",
        root / "environment.lock.yml",
        root / "conda-osx-arm64.lock.txt",
        root / "requirements-lock.txt",
        primary_dir / "run_config.json",
        reproduction_dir / "run_config.json",
        primary_dir / "validation_results.csv",
        reproduction_dir / "validation_results.csv",
        primary_dir / "validation_split_indices.csv",
        reproduction_dir / "validation_split_indices.csv",
        primary_dir / "validation_predictions_best.csv",
        reproduction_dir / "validation_predictions_best.csv",
        root / "predictions.csv",
        reproduction_dir / "predictions.csv",
        primary_dir / "stability_run_config.json",
        reproduction_dir / "stability_run_config.json",
        primary_dir / "stability_results.csv",
        reproduction_dir / "stability_results.csv",
        primary_dir / "stability_split_indices.csv",
        reproduction_dir / "stability_split_indices.csv",
        primary_dir / "stability_summary.json",
        reproduction_dir / "stability_summary.json",
    ]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.is_file()]
    if missing:
        issues.extend(f"missing required file: {path}" for path in missing)
        return {
            "status": "failed",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "tolerance": TOLERANCE,
            "issues": issues,
        }

    primary = read_json(primary_dir / "run_config.json")
    reproduction = read_json(reproduction_dir / "run_config.json")
    primary_stability_config = read_json(primary_dir / "stability_run_config.json")
    reproduction_stability_config = read_json(reproduction_dir / "stability_run_config.json")

    code_hashes = {
        "run_experiment.py": sha256_file(root / "run_experiment.py"),
        "explore_data.py": sha256_file(root / "explore_data.py"),
    }
    data_hashes = {
        name: sha256_file(root / name)
        for name in ("train_data.csv", "test_data_unlabeled.csv")
    }
    lock_names = sorted(primary.get("dependency_lock_sha256", {}))
    lock_hashes = {
        name: sha256_file(root / name)
        for name in lock_names
        if (root / name).is_file()
    }

    if primary.get("status") != "completed" or reproduction.get("status") != "completed":
        issues.append("both full-run manifests must have status=completed")
    if primary.get("candidate_count") != 60 or reproduction.get("candidate_count") != 60:
        issues.append("both full runs must evaluate exactly 60 candidates")
    primary_script_hash = primary.get("runtime_environment", {}).get("script_sha256")
    reproduction_script_hash = reproduction.get("runtime_environment", {}).get("script_sha256")
    if primary_script_hash != reproduction_script_hash:
        issues.append("full-run manifests record different run_experiment.py hashes")
    if primary_script_hash != code_hashes["run_experiment.py"]:
        issues.append("full-run manifest hash does not match current run_experiment.py")
    if primary.get("input_sha256") != reproduction.get("input_sha256"):
        issues.append("full-run manifests record different input hashes")
    if primary.get("input_sha256") != data_hashes:
        issues.append("full-run input hashes do not match current training/test CSVs")
    if primary.get("dependency_lock_sha256") != reproduction.get("dependency_lock_sha256"):
        issues.append("full-run manifests record different dependency lock hashes")
    if primary.get("dependency_lock_sha256") != lock_hashes:
        issues.append("full-run lock hashes do not match current lock files")
    if primary.get("package_versions") != reproduction.get("package_versions"):
        issues.append("package versions differ between environments")

    primary_runtime = primary.get("runtime_environment", {})
    reproduction_runtime = reproduction.get("runtime_environment", {})
    environment_names = {
        "primary": primary_runtime.get("conda_environment"),
        "reproduction": reproduction_runtime.get("conda_environment"),
    }
    if environment_names != {"primary": "project1", "reproduction": "project1_reproduce"}:
        issues.append("runs did not use the expected separate Conda environments")
    for label, runtime in (("primary", primary_runtime), ("reproduction", reproduction_runtime)):
        actual_threads = runtime.get("thread_environment_variables", {})
        if any(actual_threads.get(name) != value for name, value in THREAD_LIMITS.items()):
            issues.append(f"{label}: thread environment limits are not all set to 1")

    for label, config in (
        ("primary stability", primary_stability_config),
        ("reproduction stability", reproduction_stability_config),
    ):
        if config.get("status") != "completed":
            issues.append(f"{label}: stability manifest is not completed")
        if config.get("script_sha256") != code_hashes["run_experiment.py"]:
            issues.append(f"{label}: script hash does not match current source")
        if config.get("input_sha256") != data_hashes:
            issues.append(f"{label}: input hashes do not match current CSVs")
        if config.get("package_versions") != primary.get("package_versions"):
            issues.append(f"{label}: package versions differ from the full run")
    if primary_stability_config.get("script_sha256") != reproduction_stability_config.get(
        "script_sha256"
    ):
        issues.append("stability manifests record different script hashes")

    primary_metrics = read_csv(primary_dir / "validation_results.csv")
    reproduction_metrics = read_csv(reproduction_dir / "validation_results.csv")
    metric_difference, metric_values_compared = compare_metric_rows(
        primary_metrics,
        reproduction_metrics,
        "candidate_id",
        "60-candidate metrics",
        issues,
        check_class_f1=True,
    )
    if len(primary_metrics) != 60 or len(reproduction_metrics) != 60:
        issues.append("validation_results.csv must contain 60 rows in both runs")

    validation_split_rows = compare_rows_exact(
        read_csv(primary_dir / "validation_split_indices.csv"),
        read_csv(reproduction_dir / "validation_split_indices.csv"),
        "validation split indices",
        issues,
    )
    validation_prediction_rows = compare_rows_exact(
        read_csv(primary_dir / "validation_predictions_best.csv"),
        read_csv(reproduction_dir / "validation_predictions_best.csv"),
        "best-model validation predictions",
        issues,
    )

    primary_predictions = (root / "predictions.csv").read_bytes()
    reproduction_predictions = (reproduction_dir / "predictions.csv").read_bytes()
    prediction_bytes_equal = primary_predictions == reproduction_predictions
    if not prediction_bytes_equal:
        issues.append("final prediction files are not byte-identical")

    primary_stability = read_csv(primary_dir / "stability_results.csv")
    reproduction_stability = read_csv(reproduction_dir / "stability_results.csv")
    stability_difference, stability_values_compared = compare_metric_rows(
        primary_stability,
        reproduction_stability,
        "seed",
        "stability metrics",
        issues,
    )
    if len(primary_stability) != 5 or len(reproduction_stability) != 5:
        issues.append("stability_results.csv must contain five seed rows in both runs")
    stability_split_rows = compare_rows_exact(
        read_csv(primary_dir / "stability_split_indices.csv"),
        read_csv(reproduction_dir / "stability_split_indices.csv"),
        "stability split indices",
        issues,
    )
    primary_summary = read_json(primary_dir / "stability_summary.json")
    reproduction_summary = read_json(reproduction_dir / "stability_summary.json")
    for label, summary, output_dir in (
        ("primary", primary_summary, primary_dir),
        ("reproduction", reproduction_summary, reproduction_dir),
    ):
        if summary.get("input_sha256") != data_hashes:
            issues.append(f"{label}: stability summary input hashes differ from current CSVs")
        if summary.get("best_config_sha256") != sha256_file(output_dir / "best_config.json"):
            issues.append(f"{label}: stability summary best_config hash is invalid")
    summary_fields = ("candidate_id", "candidate", "seeds", "split", "interpretation")
    primary_summary_core = {field: primary_summary.get(field) for field in summary_fields}
    reproduction_summary_core = {
        field: reproduction_summary.get(field) for field in summary_fields
    }
    summary_difference = compare_json(
        primary_summary_core, reproduction_summary_core, "stability_summary", issues
    )
    summary_difference = max(
        summary_difference,
        compare_json(
            primary_summary.get("metrics"),
            reproduction_summary.get("metrics"),
            "stability_summary.metrics",
            issues,
        ),
    )

    for label, config in (("primary", primary), ("reproduction", reproduction)):
        if config.get("best_candidate_id") != primary.get("best_candidate_id"):
            issues.append(f"{label}: selected best candidate differs")

    return {
        "status": "passed" if not issues else "failed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": TOLERANCE,
        "environments": environment_names,
        "hashes": {
            "code": code_hashes,
            "data": data_hashes,
            "conda_locks": lock_hashes,
            "recorded_run_experiment_sha256": primary_script_hash,
            "lock_manifest_match": primary.get("dependency_lock_sha256") == lock_hashes,
            "input_manifest_match": primary.get("input_sha256") == data_hashes,
        },
        "full_runs": {
            "candidate_count": len(primary_metrics),
            "statuses": [primary.get("status"), reproduction.get("status")],
            "best_candidate_id": primary.get("best_candidate_id"),
            "metric_columns": list(METRIC_COLUMNS) + ["class_f1 per class"],
            "metric_values_compared": metric_values_compared,
            "max_absolute_metric_difference": metric_difference,
            "validation_metrics_match_within_tolerance": metric_difference <= TOLERANCE,
            "package_versions_match": primary.get("package_versions")
            == reproduction.get("package_versions"),
        },
        "validation": {
            "split_rows": len(read_csv(primary_dir / "validation_split_indices.csv")),
            "split_indices_row_by_row_identical": validation_split_rows,
            "prediction_rows": len(read_csv(primary_dir / "validation_predictions_best.csv")),
            "best_model_predictions_row_by_row_identical": validation_prediction_rows,
        },
        "final_predictions": {
            "bytes": len(primary_predictions),
            "rows": len(primary_predictions.splitlines()),
            "primary_sha256": hashlib.sha256(primary_predictions).hexdigest(),
            "reproduction_sha256": hashlib.sha256(reproduction_predictions).hexdigest(),
            "byte_identical": prediction_bytes_equal,
        },
        "stability": {
            "seeds": [int(row["seed"]) for row in primary_stability],
            "metric_values_compared": stability_values_compared,
            "max_absolute_metric_difference": stability_difference,
            "stability_summary_max_absolute_difference": summary_difference,
            "metrics_match_within_tolerance": max(stability_difference, summary_difference)
            <= TOLERANCE,
            "split_indices_row_by_row_identical": stability_split_rows,
            "ignored_timing_columns": sorted(TIMING_COLUMNS),
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing results/ and results_reproduced/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON report path; by default the report is printed to stdout",
    )
    args = parser.parse_args()
    report = build_report(args.root.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
