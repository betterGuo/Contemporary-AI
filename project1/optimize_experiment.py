#!/usr/bin/env python3
"""Run the separately planned, leakage-safe word/character optimization study.

This script is intentionally independent from the frozen baseline runner. It
reads only the labelled CSV during screening and paired validation; the
unlabelled test texts are opened only after the final candidate is locked and
the five-seed comparison is complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import statistics
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "optimization_plan.json"
BASELINE_CONFIG = {"max_features": 20000, "min_df": 1, "ngram_range": [1, 1], "C": 1.0}
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(value), ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple)) else json_safe(value)
                             for key, value in row.items()})


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_project_modules():
    """Reuse the frozen runner's numerical dependencies and metric functions."""
    import run_experiment as baseline

    baseline.load_dependencies()
    import scipy.sparse as sparse
    from sklearn.preprocessing import normalize

    return baseline, sparse, normalize


def read_training_csv(data_dir: Path, pd: Any, np: Any) -> tuple[list[str], Any]:
    path = data_dir / "train_data.csv"
    frame = pd.read_csv(path)
    if {"text", "target"} - set(frame.columns):
        raise ValueError("train_data.csv must contain 'text' and 'target' columns")
    if frame.empty or frame["text"].isna().any() or frame["target"].isna().any():
        raise ValueError("Training data is empty or contains missing text/target values")
    text = frame["text"].astype(str)
    if text.str.strip().eq("").any():
        raise ValueError("Training data contains empty or whitespace-only text")
    numeric = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Training targets must be finite integer IDs")
    labels = numeric.astype(np.int64)
    if sorted(np.unique(labels).tolist()) != list(range(10)):
        raise ValueError("Training labels must contain exactly IDs 0 through 9")
    return text.tolist(), labels


def read_test_texts_at_final_fit(data_dir: Path, pd: Any) -> list[str]:
    """This is the only function that reads test text content."""
    frame = pd.read_csv(data_dir / "test_data_unlabeled.csv")
    if "text" not in frame or frame.empty or frame["text"].isna().any():
        raise ValueError("test_data_unlabeled.csv must contain nonempty, nonmissing text")
    text = frame["text"].astype(str)
    if text.str.strip().eq("").any():
        raise ValueError("test_data_unlabeled.csv contains empty or whitespace-only text")
    return text.tolist()


def make_word_spec(max_features: int | None, min_df: int, C: float) -> dict[str, Any]:
    return {
        "representation": "word_only",
        "word": {"max_features": max_features, "min_df": min_df, "ngram_range": [1, 1]},
        "C": float(C),
    }


def candidate_id(spec: dict[str, Any], prefix: str = "word") -> str:
    word = spec.get("word", {})
    if spec["representation"] == "word_only":
        mf = "all" if word.get("max_features") is None else str(word["max_features"] // 1000) + "k"
        c = f"{spec['C']:g}".replace(".", "p")
        return f"{prefix}_mf{mf}_md{word['min_df']}_C{c}"
    return str(spec["name"])


def vectorizer_params(vectorizer: Any) -> dict[str, Any]:
    return json_safe(vectorizer.get_params(deep=True))


def csr_storage(matrix: Any, sparse: Any) -> dict[str, Any]:
    csr = matrix.tocsr(copy=False) if not sparse.isspmatrix_csr(matrix) else matrix
    return {
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "nnz": int(csr.nnz),
        "data_nbytes": int(csr.data.nbytes),
        "indices_nbytes": int(csr.indices.nbytes),
        "indptr_nbytes": int(csr.indptr.nbytes),
        "csr_storage_nbytes": int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes),
        "meaning": "CSR feature-matrix storage, not process RSS",
    }


def class_metrics(y_true: Any, y_pred: Any, baseline: Any) -> dict[str, Any]:
    report = baseline.classification_report(
        y_true, y_pred, labels=list(range(10)), target_names=[str(i) for i in range(10)],
        output_dict=True, zero_division=0,
    )
    return {
        str(label): {
            "precision": float(report[str(label)]["precision"]),
            "recall": float(report[str(label)]["recall"]),
            "f1": float(report[str(label)]["f1-score"]),
            "support": int(report[str(label)]["support"]),
        }
        for label in range(10)
    }


def summarize_metrics(y_true: Any, y_pred: Any, baseline: Any) -> dict[str, Any]:
    return {
        "accuracy": float(baseline.accuracy_score(y_true, y_pred)),
        "macro_f1": float(baseline.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(baseline.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_metrics": class_metrics(y_true, y_pred, baseline),
        "confusion_matrix": baseline.confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist(),
    }


def runtime_manifest(baseline: Any, data_dir: Path, plan_sha: str, script_sha: str) -> dict[str, Any]:
    environment = baseline.runtime_environment()
    environment.update({
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": script_sha,
        "frozen_baseline_script_path": str((HERE / "run_experiment.py").resolve()),
        "frozen_baseline_script_sha256": sha256_file(HERE / "run_experiment.py"),
        "optimization_plan_path": str(PLAN_PATH.resolve()),
        "optimization_plan_sha256": plan_sha,
    })
    return {
        "package_versions": baseline.package_versions(),
        "installed_distributions": baseline.all_installed_distributions(),
        "runtime_environment": environment,
        "input_sha256": baseline.input_checksums(data_dir),
        "dependency_lock_sha256": baseline.dependency_lock_checksums(data_dir),
    }


def build_branch_matrices(kind: str, seed: int, word_spec: dict[str, Any] | None,
                          train_texts: list[str], val_texts: list[str], baseline: Any,
                          cache: dict[tuple[Any, ...], dict[str, Any]], sparse: Any) -> dict[str, Any]:
    """Fit a word or char branch on the seed's training split only; cache identical branches."""
    if kind == "word":
        assert word_spec is not None
        # The caller passes the normalized word-branch parameter object itself.
        word = word_spec
        key = (kind, seed, word["max_features"], word["min_df"])
        vectorizer = baseline.TfidfVectorizer(
            max_features=word["max_features"], ngram_range=(1, 1), min_df=word["min_df"],
            sublinear_tf=True, strip_accents="unicode", lowercase=True, dtype=baseline.np.float64,
        )
    else:
        key = (kind, seed, 80000, 2, (3, 5))
        vectorizer = baseline.TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=80000, min_df=2,
            sublinear_tf=True, strip_accents="unicode", lowercase=True, dtype=baseline.np.float64,
        )
    if key in cache:
        return {**cache[key], "cache_hit": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        X_train = vectorizer.fit_transform(train_texts)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        X_val = vectorizer.transform(val_texts)
        transform_seconds = time.perf_counter() - start
    record = {
        "X_train": X_train.tocsr(), "X_val": X_val.tocsr(), "vectorizer": vectorizer,
        "fit_seconds": float(fit_seconds), "transform_seconds": float(transform_seconds),
        "parameters": vectorizer_params(vectorizer),
        "warnings": baseline._warning_records(caught),
        "actual_features": int(X_train.shape[1]),
        "cache_key": list(key),
        "cache_hit": False,
    }
    cache[key] = record
    return record


def candidate_matrices(spec: dict[str, Any], seed: int, train_texts: list[str], val_texts: list[str],
                       baseline: Any, sparse: Any, normalize: Any,
                       cache: dict[tuple[Any, ...], dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
    representation = spec["representation"]
    word_branch = char_branch = None
    started = time.perf_counter()
    if representation in ("word_only", "fusion"):
        word_branch = build_branch_matrices("word", seed, spec["word"], train_texts, val_texts,
                                            baseline, cache, sparse)
    if representation in ("char_only", "fusion"):
        char_branch = build_branch_matrices("char", seed, None, train_texts, val_texts,
                                            baseline, cache, sparse)
    combination_seconds = 0.0
    if representation == "word_only":
        X_train, X_val = word_branch["X_train"], word_branch["X_val"]
        branches = {"word": word_branch}
    elif representation == "char_only":
        X_train, X_val = char_branch["X_train"], char_branch["X_val"]
        branches = {"char": char_branch}
    elif representation == "fusion":
        ww, cw = float(spec["word_weight"]), float(spec["char_weight"])
        combine_started = time.perf_counter()
        X_train = sparse.hstack((word_branch["X_train"] * ww, char_branch["X_train"] * cw), format="csr")
        X_val = sparse.hstack((word_branch["X_val"] * ww, char_branch["X_val"] * cw), format="csr")
        X_train = normalize(X_train, norm="l2", axis=1, copy=False).tocsr()
        X_val = normalize(X_val, norm="l2", axis=1, copy=False).tocsr()
        combination_seconds = time.perf_counter() - combine_started
        branches = {"word": word_branch, "char": char_branch}
    else:
        raise ValueError(f"Unknown representation {representation!r}")
    details = {
        "representation": representation,
        "feature_prep_wall_seconds_including_cache_lookup": float(time.perf_counter() - started),
        "fusion_combine_normalize_seconds": float(combination_seconds),
        "branches": {
            name: {
                "parameters": branch["parameters"],
                "cache_key": branch["cache_key"],
                "fit_seconds": branch["fit_seconds"],
                "transform_seconds": branch["transform_seconds"],
                "actual_features": branch["actual_features"],
                "cache_hit": branch["cache_hit"],
                "warnings": branch["warnings"],
                "train_csr": csr_storage(branch["X_train"], sparse),
                "validation_csr": csr_storage(branch["X_val"], sparse),
            }
            for name, branch in branches.items()
        },
        "train_csr": csr_storage(X_train, sparse),
        "validation_csr": csr_storage(X_val, sparse),
        "combination_parameters": (
            {"word_weight": float(spec["word_weight"]), "char_weight": float(spec["char_weight"]),
             "branch_normalization": "L2", "combined_normalization": "L2"}
            if representation == "fusion" else {"post_branch_normalization": "skipped to preserve branch matrix exactly"}
        ),
    }
    return X_train, X_val, details


def evaluate_candidate(spec: dict[str, Any], seed: int, split: dict[str, Any], texts: list[str], labels: Any,
                       baseline: Any, sparse: Any, normalize: Any,
                       feature_cache: dict[tuple[Any, ...], dict[str, Any]],
                       fit_counter: list[int]) -> tuple[dict[str, Any], Any]:
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    train_texts = [texts[int(i)] for i in train_idx]
    val_texts = [texts[int(i)] for i in val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    started = time.perf_counter()
    X_train, X_val, feature_details = candidate_matrices(
        spec, seed, train_texts, val_texts, baseline, sparse, normalize, feature_cache,
    )
    model = baseline.LinearSVC(C=float(spec["C"]), random_state=int(seed), max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_counter[0] += 1
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        train_predict_start = time.perf_counter()
        train_prediction = model.predict(X_train)
        train_predict_seconds = time.perf_counter() - train_predict_start
        val_predict_start = time.perf_counter()
        val_prediction = model.predict(X_val)
        val_predict_seconds = time.perf_counter() - val_predict_start
    coef_bytes = int(model.coef_.nbytes)
    intercept_bytes = int(model.intercept_.nbytes)
    warning_records = [*feature_details["branches"].get("word", {}).get("warnings", []),
                       *feature_details["branches"].get("char", {}).get("warnings", []),
                       *baseline._warning_records(caught)]
    branch_cold_seconds = sum(
        branch["fit_seconds"] + branch["transform_seconds"]
        for branch in feature_details["branches"].values()
    )
    cold_feature_seconds = float(branch_cold_seconds + feature_details["fusion_combine_normalize_seconds"])
    cold_total_seconds = float(cold_feature_seconds + fit_seconds + train_predict_seconds + val_predict_seconds)
    row = {
        "candidate_id": spec["candidate_id"], "name": spec.get("name", spec["candidate_id"]),
        "representation": spec["representation"], "seed": int(seed), "status": "completed",
        "spec": spec, "vectorizer_parameters": {
            name: branch["parameters"] for name, branch in feature_details["branches"].items()
        },
        "combination_parameters": feature_details["combination_parameters"],
        "estimator_parameters": json_safe(model.get_params(deep=True)),
        "feature_details": feature_details,
        "train_metrics": summarize_metrics(y_train, train_prediction, baseline),
        "validation_metrics": summarize_metrics(y_val, val_prediction, baseline),
        "fit_seconds": float(fit_seconds), "train_prediction_seconds": float(train_predict_seconds),
        "validation_prediction_seconds": float(val_predict_seconds),
        "feature_prep_wall_seconds_including_cache_lookup": feature_details["feature_prep_wall_seconds_including_cache_lookup"],
        "cold_feature_seconds_estimate": cold_feature_seconds,
        "cold_total_seconds_estimate": cold_total_seconds,
        "cold_time_estimate_basis": "Sum of the first measured fit+transform time for each cached branch in this split, plus this candidate's combine/normalize and estimator timings; reconstructed cold path estimate, not a separate wall-clock run.",
        "total_candidate_seconds": float(time.perf_counter() - started),
        "estimator_fit_attempted": True,
        "coef_nbytes": coef_bytes, "intercept_nbytes": intercept_bytes,
        "parameter_count": int(model.coef_.size + model.intercept_.size),
        "warnings": warning_records,
        "convergence_warning": any(item["category"] == "ConvergenceWarning" for item in warning_records),
    }
    return row, val_prediction


def matrix_split_rows(seed: int, split: dict[str, Any], labels: Any) -> list[dict[str, Any]]:
    return ([{"seed": seed, "original_index": int(i), "split": "train", "target": int(labels[i])}
             for i in split["train_idx"]]
            + [{"seed": seed, "original_index": int(i), "split": "validation", "target": int(labels[i])}
               for i in split["val_idx"]])


def record_predictions(prediction_rows: list[dict[str, Any]], stage: str, arm: str,
                       spec: dict[str, Any], seed: int, split: dict[str, Any],
                       labels: Any, predictions: Any) -> None:
    for original_index, true_label, prediction in zip(split["val_idx"], labels[split["val_idx"]], predictions):
        prediction_rows.append({
            "stage": stage, "arm": arm, "candidate_id": spec["candidate_id"], "seed": int(seed),
            "original_index": int(original_index), "target": int(true_label),
            "prediction": int(prediction), "correct": bool(true_label == prediction),
        })


def select_best(candidate_ids: list[str], result_by_seed42: dict[str, dict[str, Any]]) -> str:
    winner: str | None = None
    for candidate in candidate_ids:
        result = result_by_seed42.get(candidate)
        if not result or result["status"] != "completed":
            continue
        if winner is None or result["validation_metrics"]["macro_f1"] > result_by_seed42[winner]["validation_metrics"]["macro_f1"]:
            winner = candidate
    if winner is None:
        raise RuntimeError("No successful candidate remains for the predeclared selection step")
    return winner


def copy_alias_result(source: dict[str, Any], stage: str, arm: str) -> dict[str, Any]:
    return {**source, "stage": stage, "arm": arm, "reused_evaluation": True}


def error_analysis(texts: list[str], labels: Any, split: dict[str, Any],
                   baseline_pred: Any, final_pred: Any, baseline_metrics: dict[str, Any],
                   final_metrics: dict[str, Any], np: Any) -> dict[str, Any]:
    val_idx = split["val_idx"]
    lengths = np.asarray([len(text) for text in texts], dtype=float)
    cut_points = np.quantile(lengths, [0.25, 0.50, 0.75]).tolist()
    true = labels[val_idx]
    baseline_correct = baseline_pred == true
    final_correct = final_pred == true
    buckets = np.searchsorted(cut_points, lengths[val_idx], side="right")
    by_quartile: list[dict[str, Any]] = []
    for bucket in range(4):
        mask = buckets == bucket
        if not mask.any():
            continue
        baseline_subset = baseline_pred[mask]
        final_subset = final_pred[mask]
        true_subset = true[mask]
        by_quartile.append({
            "quartile": f"Q{bucket + 1}", "rows": int(mask.sum()),
            "baseline_accuracy": float((baseline_subset == true_subset).mean()),
            "final_accuracy": float((final_subset == true_subset).mean()),
            "accuracy_delta": float((final_subset == true_subset).mean() - (baseline_subset == true_subset).mean()),
        })
    corrected = (~baseline_correct) & final_correct
    newly_wrong = baseline_correct & (~final_correct)

    def examples(mask: Any, label: str) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for position in np.flatnonzero(mask)[:15]:
            source = texts[int(val_idx[position])]
            subject = next((line.strip()[8:].strip() for line in source.splitlines()
                            if line.strip().lower().startswith("subject:")), "")
            selected.append({
                "original_index": int(val_idx[position]), "subject": subject,
                "target": int(true[position]), "baseline_prediction": int(baseline_pred[position]),
                "final_prediction": int(final_pred[position]), "text_chars": int(len(source)),
                "change": label,
            })
        return selected

    label_metrics: dict[str, Any] = {}
    for label in (2, 7):
        label_metrics[str(label)] = {
            "baseline": baseline_metrics["class_metrics"][str(label)],
            "final": final_metrics["class_metrics"][str(label)],
        }
    return {
        "seed": 42,
        "validation_rows": int(len(val_idx)),
        "text_length_quartile_cut_points_from_labeled_corpus_chars": cut_points,
        "validation_accuracy_by_length_quartile": by_quartile,
        "class_2_and_7_metrics": label_metrics,
        "baseline_errors": int((~baseline_correct).sum()),
        "final_errors": int((~final_correct).sum()),
        "corrected_baseline_errors": int(corrected.sum()),
        "new_errors_on_previously_correct_rows": int(newly_wrong.sum()),
        "wrong_both": int(((~baseline_correct) & (~final_correct)).sum()),
        "changed_prediction_but_wrong_both": int(((~baseline_correct) & (~final_correct) & (baseline_pred != final_pred)).sum()),
        "corrected_examples": examples(corrected, "baseline_wrong_final_correct"),
        "new_error_examples": examples(newly_wrong, "baseline_correct_final_wrong"),
    }


def paired_summary(stage3_rows: list[dict[str, Any]], seeds: list[int], arms: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    by_seed_arm = {(row["seed"], row["arm"]): row for row in stage3_rows}
    missing = [(seed, arm) for seed in seeds for arm in arms
               if (seed, arm) not in by_seed_arm
               or not by_seed_arm[(seed, arm)].get("validation_metrics")]
    if missing:
        raise RuntimeError(f"Cannot summarize paired stage-3 results; failed/missing arm-seed pairs: {missing}")
    for arm in arms:
        deltas: list[float] = []
        accuracy_deltas: list[float] = []
        arm_f1_values: list[float] = []
        arm_accuracy_values: list[float] = []
        for seed in seeds:
            baseline_metrics = by_seed_arm[(seed, "baseline")]["validation_metrics"]
            arm_metrics = by_seed_arm[(seed, arm)]["validation_metrics"]
            baseline_value = baseline_metrics["macro_f1"]
            arm_value = arm_metrics["macro_f1"]
            delta = float(arm_value - baseline_value)
            deltas.append(delta)
            accuracy_deltas.append(float(arm_metrics["accuracy"] - baseline_metrics["accuracy"]))
            arm_f1_values.append(float(arm_value))
            arm_accuracy_values.append(float(arm_metrics["accuracy"]))
            rows.append({"seed": seed, "arm": arm, "reference_arm": "baseline",
                         "arm_macro_f1": arm_value, "baseline_macro_f1": baseline_value,
                         "paired_delta_macro_f1": delta,
                         "paired_delta_accuracy": accuracy_deltas[-1]})
        summaries[arm] = {
            "macro_f1_mean": float(statistics.mean(arm_f1_values)),
            "macro_f1_sample_sd": float(statistics.stdev(arm_f1_values)) if len(arm_f1_values) > 1 else 0.0,
            "accuracy_mean": float(statistics.mean(arm_accuracy_values)),
            "accuracy_sample_sd": float(statistics.stdev(arm_accuracy_values)) if len(arm_accuracy_values) > 1 else 0.0,
            "delta_mean": float(statistics.mean(deltas)),
            "delta_sample_sd": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
            "paired_accuracy_delta_mean": float(statistics.mean(accuracy_deltas)),
            "paired_accuracy_delta_sample_sd": float(statistics.stdev(accuracy_deltas)) if len(accuracy_deltas) > 1 else 0.0,
            "deltas_by_seed": deltas,
            "interpretation": "Descriptive paired split sensitivity only; folds overlap; not a confidence interval or significance test.",
        }
    return rows, summaries


def make_plots(output_dir: Path, candidate_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]],
               paired: dict[str, Any], final_metrics: dict[str, Any], plot_status: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        plot_status["errors"].append({"plot": "all", "error": f"{type(exc).__name__}: {exc}"})
        return

    def save(name: str, fig: Any) -> None:
        try:
            fig.tight_layout()
            fig.savefig(output_dir / name, dpi=160, bbox_inches="tight")
            plt.close(fig)
            plot_status["generated"].append(name)
        except Exception as exc:
            plt.close(fig)
            plot_status["errors"].append({"plot": name, "error": f"{type(exc).__name__}: {exc}"})

    successful = []
    seen_ids: set[str] = set()
    for row in candidate_rows:
        candidate = row.get("candidate_id")
        if (row.get("stage") == "stage1" and row.get("status") == "completed"
                and candidate not in seen_ids):
            successful.append(row)
            seen_ids.add(str(candidate))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(successful))
    ax.bar(x, [row["validation_metrics"]["macro_f1"] for row in successful], color="#4169a1")
    ax.set_xticks(x, [row["candidate_id"].replace("word_", "") for row in successful], rotation=35, ha="right")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Stage 1: fixed-seed word TF-IDF screening (train only fit)")
    save("stage1_word_search.png", fig)

    successful2 = [row for row in stage2_rows if row.get("status") == "completed"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(successful2))
    ax.bar(x, [row["validation_metrics"]["macro_f1"] for row in successful2], color="#4d9b75")
    ax.set_xticks(x, [row["arm"] for row in successful2], rotation=25, ha="right")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Stage 2: paired word/character representation ablation (seed 42)")
    save("stage2_representation_ablation.png", fig)

    names = [name for name in paired if name != "baseline"]
    means = [paired[name]["delta_mean"] for name in names]
    sds = [paired[name]["delta_sample_sd"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sds, capsize=4, color="#ba7945")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Paired Macro-F1 delta vs baseline (mean ± sample SD)")
    ax.set_title("Stage 3: five matched split differences (descriptive only)")
    save("stage3_paired_deltas.png", fig)

    cm = np.asarray(final_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label ID"); ax.set_ylabel("True label ID")
    ax.set_title("Seed-42 validation confusion matrix: locked final winner")
    fig.colorbar(image, ax=ax)
    save("final_winner_confusion_matrix.png", fig)


def format_metric_row(stage: str, arm: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage, "arm": arm, "candidate_id": row.get("candidate_id"),
        "seed": row.get("seed"), "status": row.get("status"),
        "representation": row.get("representation"), "C": row.get("spec", {}).get("C"),
        "max_features": row.get("spec", {}).get("word", {}).get("max_features"),
        "min_df": row.get("spec", {}).get("word", {}).get("min_df"),
        "train_accuracy": row.get("train_metrics", {}).get("accuracy"),
        "train_macro_f1": row.get("train_metrics", {}).get("macro_f1"),
        "validation_accuracy": row.get("validation_metrics", {}).get("accuracy"),
        "validation_macro_f1": row.get("validation_metrics", {}).get("macro_f1"),
        "validation_weighted_f1": row.get("validation_metrics", {}).get("weighted_f1"),
        "fit_seconds": row.get("fit_seconds"),
        "train_prediction_seconds": row.get("train_prediction_seconds"),
        "validation_prediction_seconds": row.get("validation_prediction_seconds"),
        "feature_prep_wall_seconds_including_cache_lookup": row.get("feature_prep_wall_seconds_including_cache_lookup"),
        "cold_feature_seconds_estimate": row.get("cold_feature_seconds_estimate"),
        "cold_total_seconds_estimate": row.get("cold_total_seconds_estimate"),
        "cold_time_estimate_basis": row.get("cold_time_estimate_basis"),
        "total_candidate_seconds": row.get("total_candidate_seconds"),
        "actual_features": row.get("feature_details", {}).get("train_csr", {}).get("shape", [None, None])[1],
        "train_csr_storage_nbytes": row.get("feature_details", {}).get("train_csr", {}).get("csr_storage_nbytes"),
        "validation_csr_storage_nbytes": row.get("feature_details", {}).get("validation_csr", {}).get("csr_storage_nbytes"),
        "coef_nbytes": row.get("coef_nbytes"), "intercept_nbytes": row.get("intercept_nbytes"),
        "parameter_count": row.get("parameter_count"),
        "warnings": row.get("warnings", []),
        "error": row.get("error"),
    }


def run(args: argparse.Namespace) -> int:
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if output_dir == data_dir or output_dir == HERE:
        raise ValueError("Optimization output must be an independent directory, not the data or project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log = log_path.open("x", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log)
    sys.stderr = Tee(original_stderr, log)
    manifest: dict[str, Any] = {}
    metric_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fit_attempts = [0]
    try:
        baseline, sparse, normalize = load_project_modules()
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        plan_sha, script_sha = sha256_file(PLAN_PATH), sha256_file(Path(__file__).resolve())
        if plan.get("schema_version") != 1:
            raise ValueError("Unsupported optimization_plan.json schema_version")
        immutable = runtime_manifest(baseline, data_dir, plan_sha, script_sha)
        manifest = {
            "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(data_dir), "output_dir": str(output_dir), "run_log": "run.log",
            "optimization_plan": "optimization_plan.json", **immutable,
            "test_text_content_opened": False,
            "frozen_files": {
                "run_experiment.py": sha256_file(HERE / "run_experiment.py"),
                "reproduction_check.py": sha256_file(HERE / "reproduction_check.py"),
            },
        }
        write_json(output_dir / "run_config.json", manifest)
        texts, labels = read_training_csv(data_dir, baseline.pd, baseline.np)
        lengths = [len(text) for text in texts]
        manifest["training_data_summary"] = {
            "rows": len(texts), "class_counts": {str(k): int(v) for k, v in baseline.pd.Series(labels).value_counts().sort_index().items()},
            "text_chars_min": int(min(lengths)), "text_chars_median": float(baseline.np.median(lengths)),
            "text_chars_max": int(max(lengths)),
        }
        split_seeds = [int(plan["phase12_seed"]), *[int(seed) for seed in plan["stage3"]["seeds_in_order"]]]
        splits: dict[int, dict[str, Any]] = {}
        all_split_rows: list[dict[str, Any]] = []
        for seed in split_seeds:
            indices = baseline.np.arange(len(texts))
            train_idx, val_idx = baseline.train_test_split(
                indices, test_size=float(plan["validation_size"]), random_state=seed,
                stratify=labels,
            )
            splits[seed] = {"train_idx": train_idx, "val_idx": val_idx}
            all_split_rows.extend(matrix_split_rows(seed, splits[seed], labels))
        write_csv(output_dir / "split_indices.csv", all_split_rows)
        manifest["split_policy"] = {
            "stratified": True, "validation_size": plan["validation_size"],
            "phase12_seed": plan["phase12_seed"], "phase3_seeds": plan["stage3"]["seeds_in_order"],
            "validation_is_used_for_selection": True,
        }
        write_json(output_dir / "run_config.json", manifest)
        print(f"Training rows={len(texts)}; test content unopened; output={output_dir}")
        print(f"Frozen code SHA={script_sha}; plan SHA={plan_sha}")

        feature_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        result_by_candidate_seed: dict[tuple[str, int], dict[str, Any]] = {}
        predictions_by_candidate_seed: dict[tuple[str, int], Any] = {}

        def checkpoint() -> None:
            """Persist partial candidate evidence, including if a later arm fails."""
            write_csv(output_dir / "metrics.csv", [
                format_metric_row(str(item.get("stage")), str(item.get("arm")), item)
                for item in metric_records
            ])
            write_csv(output_dir / "validation_predictions.csv", prediction_rows)
            write_json(output_dir / "optimization_results.json", {
                "status": "running", "plan_sha256": plan_sha, "script_sha256": script_sha,
                "candidate_results": metric_records,
                "candidate_evaluation_records_including_aliases": len(metric_records),
                "unique_estimator_fit_attempts": fit_attempts[0],
                "test_text_content_opened": False,
            })

        def evaluate(spec: dict[str, Any], stage: str, arm: str, seed: int) -> dict[str, Any]:
            key = (spec["candidate_id"], seed)
            if key in result_by_candidate_seed:
                source = result_by_candidate_seed[key]
                alias = copy_alias_result(source, stage, arm)
                alias["seed"] = int(seed)
                alias["spec"] = spec
                metric_records.append(alias)
                if key in predictions_by_candidate_seed:
                    record_predictions(prediction_rows, stage, arm, spec, seed, splits[seed], labels,
                                       predictions_by_candidate_seed[key])
                checkpoint()
                return alias
            split = splits[seed]
            print(f"[{stage}] {arm}: {spec['candidate_id']} seed={seed}")
            fit_attempts_before = fit_attempts[0]
            try:
                result, val_prediction = evaluate_candidate(
                    spec, seed, split, texts, labels, baseline, sparse, normalize, feature_cache,
                    fit_attempts,
                )
            except Exception as exc:
                result = {"candidate_id": spec["candidate_id"], "name": arm,
                          "representation": spec["representation"], "seed": seed,
                          "status": "failed", "spec": spec,
                          "error": f"{type(exc).__name__}: {exc}", "warnings": [],
                          "estimator_fit_attempted": fit_attempts[0] > fit_attempts_before}
                val_prediction = None
                print(f"  FAILED: {result['error']}")
            else:
                predictions_by_candidate_seed[key] = val_prediction
                record_predictions(prediction_rows, stage, arm, spec, seed, split, labels, val_prediction)
                m = result["validation_metrics"]
                print(f"  train Macro-F1={result['train_metrics']['macro_f1']:.6f}; "
                      f"validation Macro-F1={m['macro_f1']:.6f}, accuracy={m['accuracy']:.6f}; "
                      f"fit={result['fit_seconds']:.3f}s warnings={len(result['warnings'])}")
            result_by_candidate_seed[key] = result
            result["stage"] = stage
            result["arm"] = arm
            metric_records.append(result)
            checkpoint()
            return result

        def word_id(max_features: int | None, min_df: int, C: float) -> str:
            return candidate_id(make_word_spec(max_features, min_df, C))

        # Stage 1A: vocabulary-size screen.
        stage1_specs: list[dict[str, Any]] = []
        A_specs: list[dict[str, Any]] = []
        for max_features in plan["stage1"]["steps"][0]["max_features_in_order"]:
            spec = make_word_spec(max_features, 1, 1.0)
            spec["candidate_id"] = word_id(max_features, 1, 1.0)
            if spec["candidate_id"] not in {item["candidate_id"] for item in stage1_specs}:
                stage1_specs.append(spec); A_specs.append(spec)
        A_ids = [item["candidate_id"] for item in A_specs]
        for spec in A_specs:
            evaluate(spec, "stage1", "A_max_features", 42)
        A_best = select_best(A_ids, {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})

        # Stage 1B: min_df values at A's winning vocabulary size; min_df=1 reuses A.
        A_spec = next(item for item in stage1_specs if item["candidate_id"] == A_best)
        B_specs: list[dict[str, Any]] = []
        for min_df in plan["stage1"]["steps"][1]["min_df_in_order"]:
            spec = make_word_spec(A_spec["word"]["max_features"], int(min_df), 1.0)
            spec["candidate_id"] = word_id(spec["word"]["max_features"], int(min_df), 1.0)
            if spec["candidate_id"] not in {item["candidate_id"] for item in stage1_specs}:
                stage1_specs.append(spec)
            B_specs.append(spec)
        for spec in B_specs:
            evaluate(spec, "stage1", "B_min_df", 42)
        B_best = select_best([item["candidate_id"] for item in B_specs],
                             {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})

        # Stage 1C: C values at B's winning vocabulary/min_df; C=1 reuses A/B.
        B_spec = next(item for item in stage1_specs if item["candidate_id"] == B_best)
        C_specs: list[dict[str, Any]] = []
        for C in plan["stage1"]["steps"][2]["C_in_order"]:
            spec = make_word_spec(B_spec["word"]["max_features"], B_spec["word"]["min_df"], float(C))
            spec["candidate_id"] = word_id(spec["word"]["max_features"], spec["word"]["min_df"], float(C))
            if spec["candidate_id"] not in {item["candidate_id"] for item in stage1_specs}:
                stage1_specs.append(spec)
            C_specs.append(spec)
        for spec in C_specs:
            evaluate(spec, "stage1", "C_svm_C", 42)
        C_best = select_best([item["candidate_id"] for item in C_specs],
                             {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})
        step_winners = list(dict.fromkeys([A_best, B_best, C_best]))
        best_word_id = select_best(step_winners,
                                   {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})
        best_word_spec = next(item for item in stage1_specs if item["candidate_id"] == best_word_id)
        print(f"Stage 1 winners A={A_best}, B={B_best}, C={C_best}; locked word={best_word_id}")

        # Include the frozen baseline as an explicit search candidate without touching its artifacts.
        baseline_id = word_id(BASELINE_CONFIG["max_features"], BASELINE_CONFIG["min_df"], BASELINE_CONFIG["C"])
        baseline_spec = make_word_spec(20000, 1, 1.0)
        baseline_spec.update({"candidate_id": baseline_id, "name": "baseline"})
        baseline_result = evaluate(baseline_spec, "baseline_reference", "baseline", 42)

        # Stage 2: use the selected word settings and fixed character parameters.
        stage2_specs: list[dict[str, Any]] = [best_word_spec]
        stage2_arm_specs: dict[str, dict[str, Any]] = {"word_only": best_word_spec}
        char_spec = {
            "candidate_id": "char_only", "name": "char_only", "representation": "char_only",
            "C": float(best_word_spec["C"]),
        }
        stage2_arm_specs["char_only"] = char_spec
        fusion_specs: list[dict[str, Any]] = []
        for arm in plan["stage2"]["arms_in_order"]:
            name = arm["name"]
            if name in ("word_only", "char_only"):
                continue
            spec = {
                "candidate_id": name, "name": name, "representation": "fusion",
                "word": best_word_spec["word"], "C": float(best_word_spec["C"]),
                "word_weight": float(arm["word_weight"]), "char_weight": float(arm["char_weight"]),
            }
            fusion_specs.append(spec); stage2_arm_specs[name] = spec
        stage2_records: list[dict[str, Any]] = []
        word_result = evaluate(best_word_spec, "stage2", "word_only", 42)
        stage2_records.append(copy_alias_result(word_result, "stage2", "word_only"))
        char_result = evaluate(char_spec, "stage2", "char_only", 42)
        stage2_records.append(char_result)
        for spec in fusion_specs:
            stage2_records.append(evaluate(spec, "stage2", spec["name"], 42))
        best_fusion_id = select_best([spec["candidate_id"] for spec in fusion_specs],
                                     {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})

        # Predeclare the overall seed-42 winner before stage 3. Exact ties use first appearance.
        all_search_specs: list[dict[str, Any]] = list(stage1_specs)
        if baseline_id not in {spec["candidate_id"] for spec in all_search_specs}:
            all_search_specs.append(baseline_spec)
        all_search_specs.extend([char_spec, *fusion_specs])
        unique_search_specs: dict[str, dict[str, Any]] = {}
        for spec in all_search_specs:
            unique_search_specs.setdefault(spec["candidate_id"], spec)
        all_search_specs = list(unique_search_specs.values())
        final_winner_id = select_best([spec["candidate_id"] for spec in all_search_specs],
                                      {key[0]: value for key, value in result_by_candidate_seed.items() if key[1] == 42})
        spec_by_id = {spec["candidate_id"]: spec for spec in all_search_specs}
        final_winner_spec = spec_by_id[final_winner_id]
        selection = {
            "stage1": {"A_best": A_best, "B_best": B_best, "C_best": C_best,
                       "phase1_winner_candidates_in_order": step_winners,
                       "best_word": best_word_id},
            "stage1_unique_candidate_ids_in_order": [spec["candidate_id"] for spec in stage1_specs],
            "stage2_best_fusion": best_fusion_id,
            "seed42_overall_candidates_in_order": [spec["candidate_id"] for spec in all_search_specs],
            "final_winner_locked_before_stage3": final_winner_id,
            "final_winner_spec": final_winner_spec,
            "selection_metric": "macro_f1", "tie_break": plan["tie_break"],
        }
        write_json(output_dir / "selection_manifest.json", selection)
        manifest["selection_locked_before_stage3"] = selection
        write_json(output_dir / "run_config.json", manifest)
        print(f"Stage 2 best fusion={best_fusion_id}; locked seed-42 overall winner={final_winner_id}")

        # Stage 3: four locked arms on identical new stratified splits; never retune here.
        arm_specs = {
            "baseline": baseline_spec,
            "best_word": best_word_spec,
            "char_only": char_spec,
            "best_fusion": stage2_arm_specs[best_fusion_id],
        }
        stage3_records: list[dict[str, Any]] = []
        feature_cache.clear()
        for seed in [int(value) for value in plan["stage3"]["seeds_in_order"]]:
            for arm in plan["stage3"]["arms_in_order"]:
                spec = arm_specs[arm]
                stage3_records.append(evaluate(spec, "stage3", arm, seed))
            # Paired arms share feature branches within a seed; release them before the next split.
            feature_cache.clear()
        paired_rows, paired = paired_summary(
            stage3_records, [int(value) for value in plan["stage3"]["seeds_in_order"]],
            list(plan["stage3"]["arms_in_order"]),
        )

        # Error diagnostics use only the already-evaluated seed-42 validation predictions.
        baseline_prediction = predictions_by_candidate_seed[(baseline_id, 42)]
        final_prediction = predictions_by_candidate_seed[(final_winner_id, 42)]
        baseline_val = result_by_candidate_seed[(baseline_id, 42)]["validation_metrics"]
        final_val = result_by_candidate_seed[(final_winner_id, 42)]["validation_metrics"]
        errors = error_analysis(texts, labels, splits[42], baseline_prediction, final_prediction,
                                baseline_val, final_val, baseline.np)

        # Final locked fit: this is the first point at which test text content is read.
        test_texts = read_test_texts_at_final_fit(data_dir, baseline.pd)
        manifest["test_text_content_opened"] = True
        manifest["test_text_content_opened_at_utc"] = datetime.now(timezone.utc).isoformat()
        final_spec = final_winner_spec
        full_split = {"train_idx": baseline.np.arange(len(texts)), "val_idx": baseline.np.arange(len(texts))}
        X_all, X_test, final_features = candidate_matrices(
            final_spec, 42, texts, test_texts, baseline, sparse, normalize,
            # The cached validation matrices cannot fit full data. Use fresh full-data feature cache.
            {},
        )
        final_model = baseline.LinearSVC(C=float(final_spec["C"]), random_state=42, max_iter=5000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_start = time.perf_counter(); final_model.fit(X_all, labels); final_fit_seconds = time.perf_counter() - fit_start
            train_pred_start = time.perf_counter(); full_train_pred = final_model.predict(X_all); full_train_pred_seconds = time.perf_counter() - train_pred_start
            test_pred_start = time.perf_counter(); test_predictions = final_model.predict(X_test); test_prediction_seconds = time.perf_counter() - test_pred_start
        final_warnings = [*final_features["branches"].get("word", {}).get("warnings", []),
                          *final_features["branches"].get("char", {}).get("warnings", []),
                          *baseline._warning_records(caught)]
        final_fit = {
            "candidate_id": final_winner_id, "spec": final_spec,
            "vectorizer_parameters": {name: info["parameters"] for name, info in final_features["branches"].items()},
            "combination_parameters": final_features["combination_parameters"],
            "estimator_parameters": json_safe(final_model.get_params(deep=True)),
            "train_metrics_resubstitution_only": summarize_metrics(labels, full_train_pred, baseline),
            "vectorizer_fit_and_transform_seconds": {
                name: {"fit": info["fit_seconds"], "test_transform": info["transform_seconds"]}
                for name, info in final_features["branches"].items()
            },
            "combine_normalize_seconds": final_features["fusion_combine_normalize_seconds"],
            "estimator_fit_seconds": float(final_fit_seconds),
            "train_prediction_seconds": float(full_train_pred_seconds),
            "test_prediction_seconds": float(test_prediction_seconds),
            "total_seconds": float(sum(info["fit_seconds"] + info["transform_seconds"]
                                       for info in final_features["branches"].values())
                                   + final_features["fusion_combine_normalize_seconds"]
                                   + final_fit_seconds + full_train_pred_seconds + test_prediction_seconds),
            "train_csr": final_features["train_csr"], "test_csr": final_features["validation_csr"],
            "coef_nbytes": int(final_model.coef_.nbytes), "intercept_nbytes": int(final_model.intercept_.nbytes),
            "parameter_count": int(final_model.coef_.size + final_model.intercept_.size),
            "warnings": final_warnings,
            "submission_rows": int(len(test_predictions)),
            "prediction_values": [int(v) for v in sorted(baseline.np.unique(test_predictions).tolist())],
            "submission_sha256": None,
        }
        submission_path = output_dir / "predictions.csv"
        with submission_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerows([[int(value)] for value in test_predictions])
        final_fit["submission_sha256"] = sha256_file(submission_path)
        final_fit["warnings_count"] = len(final_warnings)

        metric_rows = [format_metric_row(str(row.get("stage")), str(row.get("arm")), row)
                       for row in metric_records]
        write_csv(output_dir / "metrics.csv", metric_rows)
        write_csv(output_dir / "validation_predictions.csv", prediction_rows)
        write_csv(output_dir / "paired_deltas.csv", paired_rows)
        write_json(output_dir / "paired_summary.json", {
            "reference_arm": "baseline", "seeds": plan["stage3"]["seeds_in_order"],
            "arms": paired, "interpretation": plan["stage3"]["summary"],
        })
        write_json(output_dir / "error_analysis_seed42.json", errors)
        stage1_metric_rows = [row for row in metric_records if row.get("stage") == "stage1"]
        stage2_metric_rows = [row for row in stage2_records]
        plot_status = {"generated": [], "errors": []}
        make_plots(output_dir, stage1_metric_rows, stage2_metric_rows, paired, final_val, plot_status)
        write_json(output_dir / "plot_generation.json", plot_status)

        results_payload = {
            "status": "completed", "plan_sha256": plan_sha, "script_sha256": script_sha,
            "selection": selection,
            "candidate_results": metric_records,
            "stage3_paired_deltas": paired_rows, "stage3_paired_summary": paired,
            "seed42_error_analysis": errors,
            "final_fit": final_fit,
            "plot_generation": plot_status,
            "interpretation_limits": [
                "All stage 1/2 selection used seed-42 validation Macro-F1; it is used for selection, not an independent test.",
                "Stage 3 folds are paired by seed but overlap in examples; report descriptive mean and sample SD only, not CIs or significance.",
                "Training scores are resubstitution diagnostics, not generalization estimates.",
                "CSR byte counts estimate sparse matrix storage and are not process RSS.",
                "The unlabeled test text was first opened after the final candidate had been locked; it was used only for the final prediction.",
            ],
        }
        write_json(output_dir / "optimization_results.json", results_payload)
        final_manifest = {
            **manifest, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_evaluation_records_including_aliases": len(metric_records),
            "unique_estimator_fit_attempts": fit_attempts[0],
            "stage1_unique_candidates": len(stage1_specs),
            "stage2_new_candidates": 1 + len(fusion_specs),
            "planned_stage3_arm_evaluations": len(stage3_records),
            "final_winner_locked_before_stage3": final_winner_id,
            "final_submission": str(submission_path), "final_submission_sha256": final_fit["submission_sha256"],
            "final_fit_warnings": final_warnings,
            "plot_generation": plot_status,
            "artifacts": ["optimization_results.json", "metrics.csv", "validation_predictions.csv",
                          "split_indices.csv", "paired_deltas.csv", "paired_summary.json",
                          "error_analysis_seed42.json", "predictions.csv", "plot_generation.json", "run.log"],
        }
        if sha256_file(Path(__file__).resolve()) != script_sha:
            raise RuntimeError("Optimization source changed during this run; refusing completed status")
        write_json(output_dir / "run_config.json", final_manifest)
        print(f"Completed. Locked winner={final_winner_id}; submission={submission_path}; "
              f"SHA256={final_fit['submission_sha256']}")
        print("Root predictions.csv and frozen baseline artifacts were not modified.")
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                         "candidate_evaluation_records_including_aliases": len(metric_records),
                         "unique_estimator_fit_attempts": fit_attempts[0]})
        try:
            write_json(output_dir / "run_config.json", manifest)
        except Exception:
            pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE,
                        help="Project directory containing the labelled and unlabeled CSV files")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_optimized",
                        help="Independent output directory; non-empty directories are never overwritten")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
