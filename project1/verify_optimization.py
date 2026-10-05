#!/usr/bin/env python3
"""Independently compare two locked optimization runs using only stdlib."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_BASELINE_SCRIPT = "066c310e648788d5b001a4144e3f9ddf3bb7f1868a39f5cf6c03af04fdce791c"
EXPECTED_BASELINE_CHECKER = "33ad878b6750d879217dd661788f1e35aab445bc7c9d105b30d17b041edb30db"
IGNORED_KEYS = {
    "started_at_utc", "completed_at_utc", "failed_at_utc", "data_dir", "output_dir",
    "submission_path", "python_executable", "python_version", "command_argv", "command_line",
    "working_directory", "conda_prefix", "conda_environment", "threadpools", "processor",
    "platform", "release", "system", "script_path", "optimization_plan_path",
    "frozen_baseline_script_path", "filename", "filepath", "lineno", "cache_hit",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            lower = str(key).lower()
            if key in IGNORED_KEYS or "seconds" in lower or "duration" in lower or "timestamp" in lower:
                continue
            if lower.endswith("_path") or lower.endswith("_file"):
                continue
            clean[key] = scrub(item)
        return clean
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def score_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = sorted({int(row["target"]) for row in rows} | {int(row["prediction"]) for row in rows})
    total = len(rows)
    correct = sum(int(row["target"]) == int(row["prediction"]) for row in rows)
    class_scores: dict[str, Any] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(int(r["target"]) == label and int(r["prediction"]) == label for r in rows)
        fp = sum(int(r["target"]) != label and int(r["prediction"]) == label for r in rows)
        fn = sum(int(r["target"]) == label and int(r["prediction"]) != label for r in rows)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_scores[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
    return {"rows": total, "accuracy": correct / total if total else 0.0,
            "macro_f1": statistics.mean(f1s) if f1s else 0.0, "class_metrics": class_scores}


def verify(primary_dir: Path, reproduced_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    issues: list[str] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            issues.append(name)

    primary_config = read_json(primary_dir / "run_config.json")
    reproduced_config = read_json(reproduced_dir / "run_config.json")
    primary_results = read_json(primary_dir / "optimization_results.json")
    reproduced_results = read_json(reproduced_dir / "optimization_results.json")
    check("both_runs_completed", primary_config.get("status") == reproduced_config.get("status") == "completed"
          and primary_results.get("status") == reproduced_results.get("status") == "completed")

    current_baseline_hash = digest(ROOT / "run_experiment.py")
    current_old_checker_hash = digest(ROOT / "reproduction_check.py")
    current_opt_hash = digest(ROOT / "optimize_experiment.py")
    current_plan_hash = digest(ROOT / "optimization_plan.json")
    check("baseline_sources_remain_frozen",
          current_baseline_hash == EXPECTED_BASELINE_SCRIPT
          and current_old_checker_hash == EXPECTED_BASELINE_CHECKER)
    runtime_a = primary_config.get("runtime_environment", {})
    runtime_b = reproduced_config.get("runtime_environment", {})
    check("optimization_source_and_plan_hashes_match_frozen_files",
          runtime_a.get("script_sha256") == runtime_b.get("script_sha256") == current_opt_hash
          and runtime_a.get("optimization_plan_sha256") == runtime_b.get("optimization_plan_sha256") == current_plan_hash)
    check("baseline_hashes_recorded_in_both_runs",
          all(config.get("frozen_files", {}).get("run_experiment.py") == EXPECTED_BASELINE_SCRIPT
              and config.get("frozen_files", {}).get("reproduction_check.py") == EXPECTED_BASELINE_CHECKER
              for config in (primary_config, reproduced_config)))
    input_hashes = {name: digest(ROOT / name) for name in ("train_data.csv", "test_data_unlabeled.csv")}
    lock_names = primary_config.get("dependency_lock_sha256", {})
    actual_lock_hashes = {name: digest(ROOT / name) for name in lock_names if (ROOT / name).is_file()}
    check("inputs_and_lock_hashes_match_files_and_both_runs",
          primary_config.get("input_sha256") == reproduced_config.get("input_sha256") == input_hashes
          and primary_config.get("dependency_lock_sha256") == reproduced_config.get("dependency_lock_sha256") == actual_lock_hashes)
    check("same_packages_but_distinct_conda_environments",
          primary_config.get("package_versions") == reproduced_config.get("package_versions")
          and primary_config.get("installed_distributions") == reproduced_config.get("installed_distributions")
          and runtime_a.get("conda_environment") != runtime_b.get("conda_environment")
          and runtime_a.get("conda_prefix") != runtime_b.get("conda_prefix"),
          {"primary": runtime_a.get("conda_environment"), "reproduced": runtime_b.get("conda_environment")})

    check("optimization_results_equal_except_runtime_fields",
          scrub(primary_results) == scrub(reproduced_results))
    check("locked_selection_equal",
          primary_results.get("selection") == reproduced_results.get("selection"))

    split_a, split_b = read_csv(primary_dir / "split_indices.csv"), read_csv(reproduced_dir / "split_indices.csv")
    check("split_indices_equal", split_a == split_b, {"rows": len(split_a)})
    pred_a = read_csv(primary_dir / "validation_predictions.csv")
    pred_b = read_csv(reproduced_dir / "validation_predictions.csv")
    check("all_candidate_validation_predictions_equal", pred_a == pred_b,
          {"prediction_rows": len(pred_a)})

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in pred_a:
        key = (row["stage"], row["arm"], row["candidate_id"], row["seed"])
        grouped.setdefault(key, []).append(row)
    candidate_records = primary_results.get("candidate_results", [])
    record_index = {(str(r.get("stage")), str(r.get("arm")), str(r.get("candidate_id")), str(r.get("seed"))): r
                    for r in candidate_records if r.get("status") == "completed"}
    metric_issues: list[str] = []
    recomputed: dict[str, Any] = {}
    for key, rows in grouped.items():
        if key not in record_index:
            metric_issues.append(f"missing candidate result {key}")
            continue
        score = score_rows(rows)
        recorded = record_index[key].get("validation_metrics", {})
        if (not close(score["accuracy"], float(recorded.get("accuracy", float("nan"))))
                or not close(score["macro_f1"], float(recorded.get("macro_f1", float("nan"))))):
            metric_issues.append(f"validation metric mismatch {key}")
        for label, computed in score["class_metrics"].items():
            reported = recorded.get("class_metrics", {}).get(label, {})
            if (int(reported.get("support", -1)) != computed["support"]
                    or any(not close(computed[field], float(reported.get(field, float("nan"))))
                           for field in ("precision", "recall", "f1"))):
                metric_issues.append(f"per-class metric mismatch {key} label={label}")
        recomputed["|".join(key)] = score
    check("validation_scores_recomputed_from_predictions", not metric_issues,
          {"groups": len(grouped), "issues": metric_issues[:20]})
    check("paired_prediction_rows_reproduce_between_runs", pred_a == pred_b)

    storage_issues: list[str] = []
    storage_groups = 0
    for record in candidate_records:
        if record.get("status") != "completed":
            continue
        details = record.get("feature_details", {})
        for set_name in ("train_csr", "validation_csr"):
            memory = details.get(set_name, {})
            triplet = [memory.get("data_nbytes"), memory.get("indices_nbytes"), memory.get("indptr_nbytes")]
            if (len(triplet) != 3 or any(not isinstance(value, int) or value < 0 for value in triplet)
                    or sum(triplet) != memory.get("csr_storage_nbytes")):
                storage_issues.append(f"CSR storage sum mismatch {record.get('candidate_id')} {set_name}")
            shape = memory.get("shape", [0, 0])
            if len(shape) != 2 or memory.get("nnz", -1) > shape[0] * shape[1]:
                storage_issues.append(f"CSR shape/nnz mismatch {record.get('candidate_id')} {set_name}")
            storage_groups += 1
        train_shape = details.get("train_csr", {}).get("shape", [0, 0])
        feature_count = int(train_shape[1]) if len(train_shape) == 2 else 0
        expected_coef_bytes = 10 * feature_count * 8
        expected_intercept_bytes = 10 * 8
        expected_params = 10 * feature_count + 10
        if (record.get("coef_nbytes") != expected_coef_bytes
                or record.get("intercept_nbytes") != expected_intercept_bytes
                or record.get("parameter_count") != expected_params):
            storage_issues.append(f"LinearSVC coefficient size/count mismatch {record.get('candidate_id')} seed={record.get('seed')}")
    check("csr_storage_and_linear_coefficient_dimensions_verified", not storage_issues,
          {"csr_matrices_checked": storage_groups, "issues": storage_issues[:20]})

    paired_rows = read_csv(primary_dir / "paired_deltas.csv")
    stage3_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in pred_a:
        if row["stage"] == "stage3":
            stage3_groups.setdefault((row["seed"], row["arm"]), []).append(row)
    paired_issues: list[str] = []
    stage3_seeds = sorted({seed for seed, _ in stage3_groups}, key=int)
    arm_names = sorted({arm for _, arm in stage3_groups})
    independent_deltas: list[dict[str, Any]] = []
    for seed in stage3_seeds:
        baseline_score = score_rows(stage3_groups[(seed, "baseline")])
        for arm in arm_names:
            arm_score = score_rows(stage3_groups[(seed, arm)])
            delta = arm_score["macro_f1"] - baseline_score["macro_f1"]
            accuracy_delta = arm_score["accuracy"] - baseline_score["accuracy"]
            saved = next((row for row in paired_rows if row["seed"] == seed and row["arm"] == arm), None)
            if (saved is None or not close(delta, float(saved["paired_delta_macro_f1"]))
                    or not close(accuracy_delta, float(saved["paired_delta_accuracy"]))):
                paired_issues.append(f"paired delta mismatch seed={seed} arm={arm}")
            independent_deltas.append({"seed": int(seed), "arm": arm,
                                       "delta_macro_f1": delta, "delta_accuracy": accuracy_delta})
    check("paired_deltas_recomputed_per_seed", not paired_issues,
          {"rows": len(paired_rows), "issues": paired_issues})
    paired_summary = read_json(primary_dir / "paired_summary.json")
    summary_issues: list[str] = []
    for arm in arm_names:
        arm_rows = [row for row in paired_rows if row["arm"] == arm]
        f1_deltas = [float(row["paired_delta_macro_f1"]) for row in arm_rows]
        acc_deltas = [float(row["paired_delta_accuracy"]) for row in arm_rows]
        recorded = paired_summary.get("arms", {}).get(arm, {})
        if not close(statistics.mean(f1_deltas), float(recorded.get("delta_mean", float("nan")))):
            summary_issues.append(f"Macro-F1 paired mean mismatch for {arm}")
        if len(f1_deltas) > 1 and not close(statistics.stdev(f1_deltas), float(recorded.get("delta_sample_sd", float("nan")))):
            summary_issues.append(f"Macro-F1 paired sample SD mismatch for {arm}")
        if not close(statistics.mean(acc_deltas), float(recorded.get("paired_accuracy_delta_mean", float("nan")))):
            summary_issues.append(f"accuracy paired mean mismatch for {arm}")
        if len(acc_deltas) > 1 and not close(statistics.stdev(acc_deltas), float(recorded.get("paired_accuracy_delta_sample_sd", float("nan")))):
            summary_issues.append(f"accuracy paired sample SD mismatch for {arm}")
    check("paired_summary_recomputed_ddof1", not summary_issues, {"issues": summary_issues})

    final_a = primary_results.get("final_fit", {})
    final_b = reproduced_results.get("final_fit", {})
    pred_file_a, pred_file_b = primary_dir / "predictions.csv", reproduced_dir / "predictions.csv"
    sha_a, sha_b = digest(pred_file_a), digest(pred_file_b)
    lines = pred_file_a.read_text(encoding="utf-8").splitlines()
    valid_submission = (len(lines) == 2457 and all(line in {str(i) for i in range(10)} for line in lines))
    check("optimized_submission_bytes_equal_and_valid",
          sha_a == sha_b == final_a.get("submission_sha256") == final_b.get("submission_sha256")
          and valid_submission,
          {"rows": len(lines), "sha256": sha_a, "values_valid": valid_submission})
    source_text = (ROOT / "optimize_experiment.py").read_text(encoding="utf-8")
    selection_write = source_text.find('write_json(output_dir / "selection_manifest.json"')
    test_open = source_text.find("test_texts = read_test_texts_at_final_fit(data_dir, baseline.pd)")
    selection_a = read_json(primary_dir / "selection_manifest.json")
    selection_b = read_json(reproduced_dir / "selection_manifest.json")
    check("test_read_occurs_after_frozen_selection_in_source_and_run_records",
          selection_write >= 0 and test_open > selection_write
          and primary_config.get("test_text_content_opened") is True
          and reproduced_config.get("test_text_content_opened") is True
          and primary_config.get("selection_locked_before_stage3", {}).get("final_winner_locked_before_stage3")
          == reproduced_config.get("selection_locked_before_stage3", {}).get("final_winner_locked_before_stage3")
          == selection_a.get("final_winner_locked_before_stage3")
          == selection_b.get("final_winner_locked_before_stage3"))

    old_check = read_json(ROOT / "reproduction_check.json")
    baseline_submission_equal = ((ROOT / "predictions.csv").read_bytes()
                                 == (ROOT / "results" / "predictions.csv").read_bytes())
    check("frozen_baseline_reproduction_still_passes",
          old_check.get("status") == "passed" and old_check.get("issues") == [])
    check("root_submission_still_matches_frozen_baseline", baseline_submission_equal,
          {"root_sha256": digest(ROOT / "predictions.csv"),
           "baseline_sha256": digest(ROOT / "results" / "predictions.csv")})

    return {
        "status": "passed" if not issues else "failed", "issues": issues,
        "primary_dir": str(primary_dir), "reproduced_dir": str(reproduced_dir),
        "checks": checks, "recomputed_validation_scores": recomputed,
        "independent_paired_deltas": independent_deltas,
        "note": "Timing, wall-clock timestamps, cache-hit state, and environment-specific paths are excluded from byte-for-byte comparison. Paired folds are descriptive and overlapping, not independent tests.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=ROOT / "results_optimized")
    parser.add_argument("--reproduced", type=Path, default=ROOT / "results_optimized_reproduced")
    parser.add_argument("--output", type=Path, default=ROOT / "optimization_check.json")
    args = parser.parse_args()
    try:
        result = verify(args.primary.resolve(), args.reproduced.resolve())
    except Exception as exc:
        result = {"status": "failed", "issues": [f"{type(exc).__name__}: {exc}"]}
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"status": result["status"], "issues": result["issues"]}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
