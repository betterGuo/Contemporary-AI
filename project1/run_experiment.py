#!/usr/bin/env python3
"""Run the reproducible Project 1 text-classification experiment.

The default run evaluates the complete 60-candidate grid on one fixed,
stratified 80/20 split (seed 42), selects by validation Macro-F1, then refits
that locked configuration on all labelled data for the unlabeled prediction
file. TF-IDF is fitted on each training partition only.

Examples (from this directory)::

    python run_experiment.py
    python run_experiment.py --quick
    python run_experiment.py --stability-only
    python run_experiment.py --stability-only --stability-seeds 13 42 73
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, TextIO


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TARGET_NAMES = [str(i) for i in range(10)]
DEFAULT_STABILITY_SEEDS = (13, 42, 73, 101, 137)
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
}


@dataclass(frozen=True)
class Candidate:
    """One model/vectorizer setting evaluated on the fixed validation split."""

    model: str
    vectorizer: str
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    alpha: float | None = None
    C: float | None = None

    @property
    def candidate_id(self) -> str:
        ngram = f"{self.ngram_range[0]}-{self.ngram_range[1]}"
        model_parameter = (
            f"alpha{self.alpha:g}" if self.alpha is not None else f"C{self.C:g}"
        )
        return (
            f"{self.model}__mf{self.max_features}__ng{ngram}"
            f"__mindf{self.min_df}__subtf{int(self.sublinear_tf)}__{model_parameter}"
        )


def check_dependencies() -> None:
    """Check package metadata before importing large third-party libraries."""

    missing: list[str] = []
    for distribution in REQUIRED_PACKAGES:
        try:
            metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(
            "Required packages are missing: "
            + ", ".join(missing)
            + ". Activate the project's Conda environment and install its locked "
            "dependencies before running this experiment."
        )


def load_dependencies() -> None:
    """Import required packages only after the friendly metadata check."""

    check_dependencies()
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        from sklearn.feature_extraction.text import TfidfVectorizer as tfidf_vectorizer
        from sklearn.linear_model import LogisticRegression as logistic_regression
        from sklearn.metrics import (
            accuracy_score as metric_accuracy_score,
            classification_report as metric_classification_report,
            confusion_matrix as metric_confusion_matrix,
            f1_score as metric_f1_score,
        )
        from sklearn.model_selection import train_test_split as stratified_train_test_split
        from sklearn.naive_bayes import MultinomialNB as multinomial_nb
        from sklearn.svm import LinearSVC as linear_svc
    except ImportError as exc:
        raise RuntimeError(
            "A required package could not be imported after its metadata check. "
            "Recreate or repair the project's Conda environment from its lock files. "
            f"Import detail: {exc}"
        ) from exc

    globals().update(
        {
            "np": numpy_module,
            "pd": pandas_module,
            "TfidfVectorizer": tfidf_vectorizer,
            "LogisticRegression": logistic_regression,
            "accuracy_score": metric_accuracy_score,
            "classification_report": metric_classification_report,
            "confusion_matrix": metric_confusion_matrix,
            "f1_score": metric_f1_score,
            "train_test_split": stratified_train_test_split,
            "MultinomialNB": multinomial_nb,
            "LinearSVC": linear_svc,
        }
    )


def package_versions() -> dict[str, str]:
    """Return runtime and core library versions for the run manifest."""

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scikit-learn": metadata.version("scikit-learn"),
        "scipy": metadata.version("scipy"),
    }
    for package in ("matplotlib", "threadpoolctl"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def all_installed_distributions() -> dict[str, str]:
    """Capture every installed Python distribution visible to this interpreter."""

    found: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            found[name] = distribution.version
    return dict(sorted(found.items(), key=lambda item: item[0].casefold()))


def json_safe(value: Any) -> Any:
    """Convert common Python/numpy values into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_checksums(data_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(data_dir / name)
        for name in ("train_data.csv", "test_data_unlabeled.csv")
    }


def dependency_lock_checksums(data_dir: Path) -> dict[str, str]:
    lock_names = (
        "conda-lock.yml",
        "conda-lock.yaml",
        "conda-linux-64.lock",
        "conda-osx-arm64.lock",
        "conda-linux-64.lock.txt",
        "conda-osx-arm64.lock.txt",
        "conda-win-64.lock.txt",
        "environment.lock.yml",
        "environment.yml",
        "requirements-lock.txt",
    )
    return {
        name: sha256_file(data_dir / name)
        for name in lock_names
        if (data_dir / name).is_file()
    }


def runtime_environment() -> dict[str, Any]:
    thread_env_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    try:
        from threadpoolctl import threadpool_info

        threadpools: Any = threadpool_info()
    except ImportError as exc:
        threadpools = {"unavailable": str(exc)}
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "thread_environment_variables": {
            name: os.environ.get(name) for name in thread_env_names
        },
        "threadpools": threadpools,
        "command_argv": [sys.executable, *sys.argv],
        "command_line": shlex.join([sys.executable, *sys.argv]),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def load_data(data_dir: Path) -> tuple[list[str], Any, list[str]]:
    """Read and strictly validate the labelled and unlabeled CSV inputs."""

    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data_unlabeled.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected train_data.csv and test_data_unlabeled.csv in {data_dir}"
        )
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as exc:
        raise ValueError(f"Could not read input CSV files: {exc}") from exc

    if {"text", "target"} - set(train_df.columns):
        raise ValueError("train_data.csv must contain 'text' and 'target' columns")
    if "text" not in test_df.columns:
        raise ValueError("test_data_unlabeled.csv must contain a 'text' column")
    if train_df.empty:
        raise ValueError("train_data.csv contains no data rows")
    if test_df.empty:
        raise ValueError("test_data_unlabeled.csv contains no data rows")

    missing_train_text = train_df["text"].isna()
    missing_train_target = train_df["target"].isna()
    missing_test_text = test_df["text"].isna()
    if missing_train_text.any() or missing_train_target.any() or missing_test_text.any():
        raise ValueError(
            "Input data contains missing values: "
            f"train text={int(missing_train_text.sum())}, "
            f"train target={int(missing_train_target.sum())}, "
            f"test text={int(missing_test_text.sum())}"
        )

    train_text_series = train_df["text"].astype(str)
    test_text_series = test_df["text"].astype(str)
    empty_train_text = train_text_series.str.strip().eq("")
    empty_test_text = test_text_series.str.strip().eq("")
    if empty_train_text.any() or empty_test_text.any():
        raise ValueError(
            "Input data contains empty or whitespace-only text: "
            f"train={int(empty_train_text.sum())}, test={int(empty_test_text.sum())}"
        )

    numeric_targets = pd.to_numeric(train_df["target"], errors="coerce")
    target_array = numeric_targets.to_numpy(dtype=float)
    if not np.isfinite(target_array).all():
        raise ValueError("Training target values must all be finite integer IDs")
    if not np.equal(target_array, np.floor(target_array)).all():
        raise ValueError("Training target values must be integer IDs")
    labels = target_array.astype(np.int64)
    observed = sorted(np.unique(labels).tolist())
    expected = list(range(10))
    if observed != expected:
        raise ValueError(
            f"Training labels must contain exactly the IDs 0 through 9; found {observed}"
        )

    return train_text_series.tolist(), labels, test_text_series.tolist()


def data_summary(texts: list[str], labels: Any, test_texts: list[str]) -> dict[str, Any]:
    lengths = np.asarray([len(text) for text in texts], dtype=np.int64)
    test_lengths = np.asarray([len(text) for text in test_texts], dtype=np.int64)
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "train_rows": len(texts),
        "test_rows": len(test_texts),
        "num_classes": int(pd.Series(labels).nunique()),
        "classes": sorted(pd.Series(labels).unique().tolist()),
        "missing_values": 0,
        "empty_text_rows": 0,
        "duplicate_train_texts": int(len(texts) - len(set(texts))),
        "duplicate_test_texts": int(len(test_texts) - len(set(test_texts))),
        "train_test_exact_text_overlap": int(len(set(texts).intersection(test_texts))),
        "train_text_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "test_text_length": {
            "min": int(test_lengths.min()),
            "mean": float(test_lengths.mean()),
            "median": float(np.median(test_lengths)),
            "max": int(test_lengths.max()),
        },
        "class_counts": {str(key): int(value) for key, value in counts.items()},
    }


def build_candidates(quick: bool = False) -> list[Candidate]:
    """Build the 60-setting default grid, with three model families."""

    if quick:
        vectorizers = [
            (5000, (1, 1), 1, True),
            (5000, (1, 2), 1, True),
        ]
        nb_values = [0.5, 1.0]
        c_values = [1.0, 10.0]
        svm_c_values = [0.1, 1.0]
    else:
        vectorizers = [
            (max_features, ngram_range, 1, True)
            for max_features in (5000, 10000, 20000)
            for ngram_range in ((1, 1), (1, 2))
        ]
        nb_values = [0.1, 0.5, 1.0]
        c_values = [0.1, 1.0, 10.0]
        svm_c_values = [0.1, 1.0, 10.0, 100.0]

    candidates: list[Candidate] = []
    for max_features, ngram_range, min_df, sublinear_tf in vectorizers:
        for alpha in nb_values:
            candidates.append(
                Candidate(
                    model="multinomial_nb",
                    vectorizer="tfidf",
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                    alpha=alpha,
                )
            )
        for C in c_values:
            candidates.append(
                Candidate(
                    model="logistic_regression",
                    vectorizer="tfidf",
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                    C=C,
                )
            )
        for C in svm_c_values:
            candidates.append(
                Candidate(
                    model="linear_svm",
                    vectorizer="tfidf",
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                    C=C,
                )
            )
    return candidates


def make_estimator(candidate: Candidate, random_state: int = RANDOM_STATE):
    if candidate.model == "multinomial_nb":
        return MultinomialNB(alpha=float(candidate.alpha))
    if candidate.model == "logistic_regression":
        return LogisticRegression(
            C=float(candidate.C),
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    if candidate.model == "linear_svm":
        return LinearSVC(C=float(candidate.C), random_state=random_state, max_iter=5000)
    raise ValueError(f"Unknown model: {candidate.model}")


def make_vectorizer(candidate: Candidate):
    return TfidfVectorizer(
        max_features=candidate.max_features,
        ngram_range=candidate.ngram_range,
        min_df=candidate.min_df,
        sublinear_tf=candidate.sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float64,
    )


def effective_configuration(candidate: Candidate, random_state: int) -> dict[str, Any]:
    vectorizer = make_vectorizer(candidate)
    estimator = make_estimator(candidate, random_state=random_state)
    return {
        "candidate": asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "vectorizer_parameters": json_safe(vectorizer.get_params(deep=True)),
        "estimator_parameters": json_safe(estimator.get_params(deep=True)),
    }


def _warning_records(caught: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": item.filename,
            "lineno": item.lineno,
        }
        for item in caught
    ]


def evaluate_candidate(
    candidate: Candidate,
    X_train_text: list[str],
    y_train: Any,
    X_val_text: list[str],
    y_val: Any,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], Any]:
    """Fit one setting and return its exact validation metrics and predictions."""

    started = time.perf_counter()
    vectorizer = make_vectorizer(candidate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vectorizer_started = time.perf_counter()
        X_train = vectorizer.fit_transform(X_train_text)
        vectorizer_fit_seconds = time.perf_counter() - vectorizer_started
        transform_started = time.perf_counter()
        X_val = vectorizer.transform(X_val_text)
        validation_transform_seconds = time.perf_counter() - transform_started

        estimator = make_estimator(candidate, random_state=random_state)
        fit_started = time.perf_counter()
        estimator.fit(X_train, y_train)
        estimator_fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = estimator.predict(X_val)
        prediction_seconds = time.perf_counter() - predict_started
    total_seconds = time.perf_counter() - started

    report = classification_report(
        y_val,
        predictions,
        labels=np.arange(10),
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    warning_records = _warning_records(caught)
    configuration = effective_configuration(candidate, random_state)
    metrics: dict[str, Any] = {
        **asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "ngram_range": str(candidate.ngram_range),
        "num_features": int(X_train.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, predictions, average="weighted", zero_division=0)),
        "vectorizer_fit_seconds": float(vectorizer_fit_seconds),
        "validation_transform_seconds": float(validation_transform_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "total_seconds": float(total_seconds),
        "convergence_warning": any(
            item["category"] == "ConvergenceWarning" for item in warning_records
        ),
        "warnings": warning_records,
        "vectorizer_parameters": configuration["vectorizer_parameters"],
        "estimator_parameters": configuration["estimator_parameters"],
        "class_f1": {
            str(label): float(report[str(label)]["f1-score"])
            for label in range(10)
        },
    }
    return metrics, predictions


def confusion_frame(y_true: Iterable[Any], y_pred: Iterable[Any]):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in range(10)],
        columns=[f"pred_{label}" for label in range(10)],
    )


def _json_column(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)


def _save_figure(plt: Any, fig: Any, output_dir: Path, filename: str, status: dict[str, Any]) -> None:
    try:
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        status["generated"].append(filename)
    except Exception as exc:  # record plot failures without claiming success
        status["failed"][filename] = repr(exc)
    finally:
        plt.close(fig)


def _save_diagnostic_plots_impl(
    metrics_df: Any,
    best_model_table: Any,
    best_confusion: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate all shared figures and write their honest success/failure status."""

    status: dict[str, Any] = {"generated": [], "failed": {}, "skipped": []}
    try:
        import matplotlib
        from matplotlib.patches import Patch

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        status["failed"]["matplotlib_import"] = repr(exc)
        write_json(output_dir / "plot_generation.json", status)
        return status

    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        status["skipped"].append("No successful candidate rows are available for plots")
    else:
        colors = {
            "multinomial_nb": "#4C78A8",
            "logistic_regression": "#F58518",
            "linear_svm": "#54A24B",
        }

        comparison = best_model_table.sort_values("macro_f1", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        y_pos = np.arange(len(comparison))
        ax.barh(
            y_pos,
            comparison["macro_f1"],
            color=[colors.get(name, "#777777") for name in comparison["model"]],
        )
        ax.set_yticks(y_pos, comparison["model"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Validation macro-F1")
        ax.set_title("Best configuration by model")
        for row_number, value in enumerate(comparison["macro_f1"]):
            ax.text(float(value) + 0.005, row_number, f"{value:.4f}", va="center")
        _save_figure(plt, fig, output_dir, "validation_model_comparison.png", status)

        # Use one global integer position per configuration; model groups no longer
        # overplot at the same x coordinates or require dense candidate labels.
        ordered = ok.sort_values(["model", "candidate_id"], kind="stable").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        positions = np.arange(1, len(ordered) + 1)
        for model_name, group in ordered.groupby("model", sort=False):
            row_positions = group.index.to_numpy() + 1
            ax.scatter(
                row_positions,
                group["macro_f1"],
                s=30,
                color=colors.get(model_name, "#777777"),
                label=model_name,
                alpha=0.85,
            )
        ax.set_xlim(0.5, max(1.5, len(positions) + 0.5))
        ax.set_xlabel("Unique candidate position (configuration details in validation_results.csv)")
        ax.set_ylabel("Validation macro-F1")
        ax.set_title("Validation Macro-F1 for every tested configuration")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        _save_figure(plt, fig, output_dir, "validation_tuning_curves.png", status)

        # Keep the historical report filename current as well: every candidate
        # gets a distinct numerical y-position so dense labels cannot overlap.
        config_positions = np.arange(len(ordered))
        fig, ax = plt.subplots(figsize=(10, max(8, len(ordered) * 0.17)))
        ax.barh(
            config_positions,
            ordered["macro_f1"],
            color=[colors.get(name, "#777777") for name in ordered["model"]],
        )
        ax.set_yticks(config_positions, [""] * len(config_positions))
        ax.set_ylim(-0.5, len(config_positions) - 0.5)
        ax.invert_yaxis()
        ax.set_xlabel("Validation macro-F1")
        ax.set_title("Validation Macro-F1 for every tested configuration")
        ax.grid(axis="x", alpha=0.25)
        handles = [
            Patch(color=color, label=model)
            for model, color in colors.items()
            if model in set(ordered["model"])
        ]
        ax.legend(handles=handles)
        _save_figure(plt, fig, output_dir, "validation_macro_f1.png", status)

        # Single-factor C curves: each line fixes model, vocabulary cap and
        # n-gram range while varying only C on a logarithmic axis.
        c_rows = ok[ok["C"].notna()].copy()
        c_rows["C"] = pd.to_numeric(c_rows["C"], errors="coerce")
        c_models = [name for name in ("logistic_regression", "linear_svm") if name in set(c_rows["model"])]
        if c_rows.empty:
            status["skipped"].append("No C-parameter candidates are available")
        else:
            fig, axes = plt.subplots(1, max(1, len(c_models)), figsize=(7 * max(1, len(c_models)), 5), squeeze=False)
            for ax, model_name in zip(axes[0], c_models):
                model_rows = c_rows[c_rows["model"] == model_name]
                for (max_features, ngram_range), group in model_rows.groupby(
                    ["max_features", "ngram_range"], sort=True
                ):
                    group = group.sort_values("C")
                    ax.plot(
                        group["C"],
                        group["macro_f1"],
                        marker="o",
                        linewidth=1.3,
                        label=f"{int(max_features)//1000}k, {ngram_range}",
                    )
                ax.set_xscale("log")
                ax.set_xlabel("C (log scale)")
                ax.set_ylabel("Validation macro-F1")
                ax.set_title(model_name)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8, title="Fixed TF-IDF settings")
            for ax in axes[0][len(c_models):]:
                ax.set_visible(False)
            fig.suptitle("Single-factor C comparison (other settings fixed within each line)")
            _save_figure(plt, fig, output_dir, "single_factor_C.png", status)

        # Single-factor alpha curves for MultinomialNB.
        alpha_rows = ok[ok["alpha"].notna()].copy()
        alpha_rows["alpha"] = pd.to_numeric(alpha_rows["alpha"], errors="coerce")
        nb_rows = alpha_rows[alpha_rows["model"] == "multinomial_nb"]
        if nb_rows.empty:
            status["skipped"].append("No MultinomialNB alpha candidates are available")
        else:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for (max_features, ngram_range), group in nb_rows.groupby(
                ["max_features", "ngram_range"], sort=True
            ):
                group = group.sort_values("alpha")
                ax.plot(
                    group["alpha"],
                    group["macro_f1"],
                    marker="o",
                    linewidth=1.3,
                    label=f"{int(max_features)//1000}k, {ngram_range}",
                )
            ax.set_xscale("log")
            ax.set_xlabel("alpha (log scale)")
            ax.set_ylabel("Validation macro-F1")
            ax.set_title("Single-factor MultinomialNB alpha comparison")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, title="Fixed TF-IDF settings")
            _save_figure(plt, fig, output_dir, "single_factor_alpha.png", status)

        # Compare vocabulary size while holding classifier strength and other
        # vectorizer settings fixed. The two lines vary only the n-gram range.
        fixed_rows = ok.copy()
        fixed_rows["C_numeric"] = pd.to_numeric(fixed_rows["C"], errors="coerce")
        fixed_rows["alpha_numeric"] = pd.to_numeric(fixed_rows["alpha"], errors="coerce")
        fixed_rows = fixed_rows[
            (
                (fixed_rows["model"] == "multinomial_nb")
                & fixed_rows["alpha_numeric"].eq(0.5)
            )
            | (
                fixed_rows["model"].isin(["logistic_regression", "linear_svm"])
                & fixed_rows["C_numeric"].eq(1.0)
            )
        ]
        if fixed_rows.empty:
            status["skipped"].append("No fixed-strength configurations for feature comparison")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
            for ax, model_name in zip(
                axes,
                ("multinomial_nb", "logistic_regression", "linear_svm"),
            ):
                model_rows = fixed_rows[fixed_rows["model"] == model_name]
                for ngram_range, group in model_rows.groupby("ngram_range", sort=True):
                    group = group.sort_values("max_features")
                    ax.plot(
                        group["max_features"],
                        group["macro_f1"],
                        marker="o",
                        label=f"n-gram {ngram_range}",
                    )
                ax.set_title(model_name)
                ax.set_xticks([5000, 10000, 20000], ["5k", "10k", "20k"])
                ax.set_xlabel("max_features")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Validation macro-F1")
            fig.suptitle(
                "Single-factor vocabulary-size comparison; NB alpha=0.5, "
                "LR/LinearSVC C=1, min_df=1, sublinear_tf=True"
            )
            _save_figure(plt, fig, output_dir, "feature_comparison.png", status)

    if not best_confusion.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(best_confusion.to_numpy(), interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(10),
            yticks=np.arange(10),
            xlabel="Predicted label",
            ylabel="True label",
            title="Best validation confusion matrix",
        )
        threshold = best_confusion.to_numpy().max() / 2.0
        for row in range(10):
            for col in range(10):
                value = int(best_confusion.iloc[row, col])
                ax.text(
                    col,
                    row,
                    value,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
        _save_figure(plt, fig, output_dir, "confusion_matrix_best.png", status)
    else:
        status["skipped"].append("Best-model confusion matrix is empty")

    write_json(output_dir / "plot_generation.json", status)
    return status


def save_diagnostic_plots(
    metrics_df: Any,
    best_model_table: Any,
    best_confusion: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Run plot creation with an explicit failure record for every exception."""

    try:
        return _save_diagnostic_plots_impl(
            metrics_df, best_model_table, best_confusion, output_dir
        )
    except Exception as exc:
        status = {
            "generated": [],
            "failed": {"plot_generation": f"{type(exc).__name__}: {exc}"},
            "skipped": [],
        }
        write_json(output_dir / "plot_generation.json", status)
        return status


def fit_full_and_predict(
    candidate: Candidate,
    texts: list[str],
    labels: Any,
    test_texts: list[str],
    random_state: int,
) -> tuple[Any, dict[str, Any]]:
    """Refit the chosen setting on all labelled rows and predict the test set."""

    started = time.perf_counter()
    vectorizer = make_vectorizer(candidate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vectorizer_started = time.perf_counter()
        X_all = vectorizer.fit_transform(texts)
        vectorizer_fit_seconds = time.perf_counter() - vectorizer_started
        transform_started = time.perf_counter()
        X_test = vectorizer.transform(test_texts)
        test_transform_seconds = time.perf_counter() - transform_started
        estimator = make_estimator(candidate, random_state=random_state)
        fit_started = time.perf_counter()
        estimator.fit(X_all, labels)
        estimator_fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = estimator.predict(X_test)
        prediction_seconds = time.perf_counter() - predict_started
    timing = {
        "vectorizer_fit_seconds": float(vectorizer_fit_seconds),
        "test_transform_seconds": float(test_transform_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "num_features": int(X_all.shape[1]),
        "num_train_rows": int(X_all.shape[0]),
        "num_test_rows": int(X_test.shape[0]),
        "warnings": _warning_records(caught),
        "vectorizer_parameters": json_safe(vectorizer.get_params(deep=True)),
        "estimator_parameters": json_safe(estimator.get_params(deep=True)),
    }
    return predictions, timing


def _as_candidate(raw: dict[str, Any]) -> Candidate:
    values = dict(raw)
    values["ngram_range"] = tuple(values["ngram_range"])
    return Candidate(**values)


def run_stability_assessment(
    data_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
) -> None:
    """Evaluate the already selected setting over explicit stratified splits."""

    best_path = output_dir / "best_config.json"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Could not find {best_path}; run the full experiment before stability assessment"
        )
    best_config = json.loads(best_path.read_text(encoding="utf-8"))
    expected_hashes = best_config.get("input_sha256")
    current_hashes = input_checksums(data_dir)
    if expected_hashes and expected_hashes != current_hashes:
        raise ValueError(
            "Input CSV checksums differ from the run that selected best_config.json. "
            "Use the original data files or rerun the complete experiment."
        )
    if not expected_hashes:
        raise ValueError(
            "best_config.json does not contain input checksums; rerun with the repaired script first"
        )

    texts, labels, _ = load_data(data_dir)
    candidate = _as_candidate(best_config["candidate"])
    selection_seed = int(best_config.get("selected_on_seed", RANDOM_STATE))
    started_at = datetime.now(timezone.utc)
    stability_manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "candidate_id": candidate.candidate_id,
        "candidate": asdict(candidate),
        "selected_on_seed": selection_seed,
        "seeds": list(seeds),
        "split": {"test_size": VALIDATION_SIZE, "stratified": True},
        "best_config_sha256": sha256_file(best_path),
        "input_sha256": current_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "package_versions": package_versions(),
        "runtime_environment": runtime_environment(),
        "effective_configurations": [
            effective_configuration(candidate, seed) for seed in seeds
        ],
        "run_log": str(output_dir / "stability.log"),
    }
    write_json(output_dir / "stability_run_config.json", stability_manifest)
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for seed in seeds:
        indices = np.arange(len(texts))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=labels,
        )
        split_rows.extend(
            {"seed": seed, "original_index": int(index), "split": "train", "target": int(labels[index])}
            for index in train_idx
        )
        split_rows.extend(
            {
                "seed": seed,
                "original_index": int(index),
                "split": "validation",
                "target": int(labels[index]),
            }
            for index in val_idx
        )
        metrics, _predictions = evaluate_candidate(
            candidate,
            [texts[index] for index in train_idx],
            labels[train_idx],
            [texts[index] for index in val_idx],
            labels[val_idx],
            random_state=seed,
        )
        rows.append(
            {
                "seed": seed,
                "candidate_id": candidate.candidate_id,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "train_rows": metrics["train_rows"],
                "validation_rows": metrics["validation_rows"],
                "num_features": metrics["num_features"],
                "vectorizer_fit_seconds": metrics["vectorizer_fit_seconds"],
                "validation_transform_seconds": metrics["validation_transform_seconds"],
                "estimator_fit_seconds": metrics["estimator_fit_seconds"],
                "prediction_seconds": metrics["prediction_seconds"],
                "convergence_warning": metrics["convergence_warning"],
                "warnings": _json_column(metrics["warnings"]),
            }
        )
        print(
            f"stability seed={seed}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(output_dir / "stability_results.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "stability_split_indices.csv", index=False)
    summary: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "candidate": asdict(candidate),
        "seeds": list(seeds),
        "split": {
            "test_size": VALIDATION_SIZE,
            "stratified": True,
            "selection_seed": selection_seed,
            "selection_seed_matches_a_stability_split": selection_seed in seeds,
        },
        "interpretation": (
            f"This is a sensitivity check of the configuration selected using seed {selection_seed}. "
            "The configuration is not retuned. Validation partitions overlap and are "
            "not independent test sets; these scores are not an unbiased test estimate."
        ),
        "metrics": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_config_sha256": stability_manifest["best_config_sha256"],
        "input_sha256": current_hashes,
    }
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        values = stability_df[metric].astype(float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    write_json(output_dir / "stability_summary.json", summary)
    stability_manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output_dir / "stability_run_config.json", stability_manifest)


def _candidate_result_row(candidate: Candidate, random_state: int) -> dict[str, Any]:
    configuration = effective_configuration(candidate, random_state)
    return {
        **asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "ngram_range": str(candidate.ngram_range),
        "vectorizer_parameters": configuration["vectorizer_parameters"],
        "estimator_parameters": configuration["estimator_parameters"],
    }


def run_full_experiment(args: argparse.Namespace, output_dir: Path) -> None:
    data_dir = args.data_dir.resolve()
    submission_path = args.submission_path.resolve()
    started_at = datetime.now(timezone.utc)
    checksums = input_checksums(data_dir)
    texts, labels, test_texts = load_data(data_dir)
    summary = data_summary(texts, labels, test_texts)
    write_json(output_dir / "dataset_summary.json", summary)

    indices = np.arange(len(texts))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=args.seed,
        stratify=labels,
    )
    X_train_text = [texts[index] for index in train_idx]
    X_val_text = [texts[index] for index in val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    split_frame = pd.DataFrame(
        {
            "original_index": np.concatenate([train_idx, val_idx]),
            "split": ["train"] * len(train_idx) + ["validation"] * len(val_idx),
            "target": np.concatenate([y_train, y_val]),
        }
    )
    split_frame.to_csv(output_dir / "validation_split_indices.csv", index=False)

    candidates = build_candidates(args.quick)
    if not args.quick and len(candidates) != 60:
        raise RuntimeError(f"The default grid must contain 60 candidates, got {len(candidates)}")
    configurations = [effective_configuration(candidate, args.seed) for candidate in candidates]
    write_json(output_dir / "candidate_configurations.json", configurations)
    config = {
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "random_state": args.seed,
        "default_seed": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "stratified_split": True,
        "selection_metric": "macro_f1",
        "selection_tie_break": "first candidate in fixed grid order",
        "validation_split_is_used_for_model_selection": True,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "submission_path": submission_path,
        "quick_grid": args.quick,
        "candidate_count": len(candidates),
        "candidate_configurations_file": "candidate_configurations.json",
        "input_sha256": checksums,
        "dependency_lock_sha256": dependency_lock_checksums(data_dir),
        "package_versions": package_versions(),
        "installed_distributions": all_installed_distributions(),
        "runtime_environment": runtime_environment(),
        "run_log": str(output_dir / "run.log"),
    }
    write_json(output_dir / "run_config.json", config)

    print("=== Project 1 text classification experiment ===")
    print(f"Python/packages: {package_versions()}")
    print(f"Python executable: {sys.executable}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {output_dir}")
    print(
        f"Validated {len(texts)} labelled rows and {len(test_texts)} test rows; "
        f"classes={summary['classes']}"
    )
    print(
        f"Stratified split: {len(train_idx)} train / {len(val_idx)} validation; "
        f"seed={args.seed}; evaluating {len(candidates)} candidates"
    )

    metrics_rows: list[dict[str, Any]] = []
    best_by_model: dict[str, tuple[dict[str, Any], Any]] = {}
    best_overall: tuple[dict[str, Any], Any] | None = None
    for number, candidate in enumerate(candidates, start=1):
        base_row = _candidate_result_row(candidate, args.seed)
        try:
            metrics, predictions = evaluate_candidate(
                candidate,
                X_train_text,
                y_train,
                X_val_text,
                y_val,
                random_state=args.seed,
            )
        except Exception as exc:
            metrics = {
                **base_row,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": [],
            }
            print(
                f"[{number}/{len(candidates)}] {candidate.candidate_id} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            metrics["status"] = "ok"
            previous = best_by_model.get(candidate.model)
            if previous is None or metrics["macro_f1"] > previous[0]["macro_f1"]:
                best_by_model[candidate.model] = (metrics, predictions)
            if best_overall is None or metrics["macro_f1"] > best_overall[0]["macro_f1"]:
                best_overall = (metrics, predictions)
            print(
                f"[{number}/{len(candidates)}] {candidate.candidate_id}: "
                f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"vectorizer={metrics['vectorizer_fit_seconds'] + metrics['validation_transform_seconds']:.2f}s, "
                f"fit={metrics['estimator_fit_seconds']:.2f}s, "
                f"predict={metrics['prediction_seconds']:.2f}s, "
                f"warnings={len(metrics['warnings'])}"
            )
        metrics_rows.append(metrics)

    serializable_rows = []
    for row in metrics_rows:
        copied = dict(row)
        for field in ("class_f1", "warnings", "vectorizer_parameters", "estimator_parameters"):
            if field in copied and not isinstance(copied[field], str):
                copied[field] = _json_column(copied[field])
        serializable_rows.append(copied)
    metrics_df = pd.DataFrame(serializable_rows)
    if "macro_f1" in metrics_df.columns:
        metrics_df.sort_values(
            by=["status", "macro_f1", "accuracy"],
            ascending=[True, False, False],
            inplace=True,
            na_position="last",
        )
    metrics_df.to_csv(output_dir / "validation_results.csv", index=False)

    class_rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        if row.get("status") != "ok":
            continue
        for class_label, score in row["class_f1"].items():
            class_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "model": row["model"],
                    "class": int(class_label),
                    "f1": score,
                }
            )
    pd.DataFrame(class_rows).to_csv(output_dir / "validation_class_f1.csv", index=False)

    if not best_by_model:
        failures = [row for row in metrics_rows if row.get("status") == "failed"]
        preview = "; ".join(
            f"{row['candidate_id']}: {row.get('error', 'unknown error')}"
            for row in failures[:3]
        )
        raise RuntimeError(
            f"All {len(candidates)} candidate configurations failed. "
            f"See {output_dir / 'validation_results.csv'}. First failures: {preview}"
        )

    # Selection is strictly Macro-F1. Exact ties retain the first configuration
    # in the fixed candidate order, consistently for model and global winners.
    assert best_overall is not None
    best_metrics, best_val_predictions = best_overall
    best_candidate = next(
        candidate for candidate in candidates if candidate.candidate_id == best_metrics["candidate_id"]
    )
    best_model_records = [item[0] for item in best_by_model.values()]
    best_model_table = pd.DataFrame(
        [
            {
                **record,
                "class_f1": _json_column(record["class_f1"]),
                "warnings": _json_column(record["warnings"]),
                "vectorizer_parameters": _json_column(record["vectorizer_parameters"]),
                "estimator_parameters": _json_column(record["estimator_parameters"]),
            }
            for record in best_model_records
        ]
    )
    best_model_table.sort_values("macro_f1", ascending=False).to_csv(
        output_dir / "best_by_model.csv", index=False
    )
    write_json(
        output_dir / "best_config.json",
        {
            "candidate": asdict(best_candidate),
            "candidate_id": best_candidate.candidate_id,
            "selection_metric": "macro_f1",
            "metrics": best_metrics,
            "input_sha256": checksums,
            "selected_on_seed": args.seed,
        },
    )

    best_confusion = confusion_frame(y_val, best_val_predictions)
    best_confusion.to_csv(output_dir / "confusion_matrix_best.csv")
    best_report = classification_report(
        y_val,
        best_val_predictions,
        labels=np.arange(10),
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(best_report).transpose().to_csv(
        output_dir / "classification_report_best.csv"
    )
    pd.DataFrame(
        {
            "validation_row": np.arange(len(val_idx)),
            "original_index": val_idx,
            "target": y_val,
            "prediction": best_val_predictions,
            "correct": y_val == best_val_predictions,
        }
    ).to_csv(output_dir / "validation_predictions_best.csv", index=False)

    for model_name, (model_metrics, model_predictions) in best_by_model.items():
        confusion_frame(y_val, model_predictions).to_csv(
            output_dir / f"confusion_matrix_{model_name}.csv"
        )

    plot_status = save_diagnostic_plots(
        metrics_df,
        best_model_table,
        best_confusion,
        output_dir,
    )
    print(
        f"Plots generated: {plot_status['generated']}; "
        f"failed: {list(plot_status['failed'])}; skipped: {plot_status['skipped']}"
    )

    print(
        f"Best validation candidate: {best_candidate.candidate_id}; "
        f"accuracy={best_metrics['accuracy']:.4f}, macro_f1={best_metrics['macro_f1']:.4f}"
    )
    final_predictions, final_timing = fit_full_and_predict(
        best_candidate, texts, labels, test_texts, random_state=args.seed
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([[int(value)] for value in final_predictions])
    results_submission = output_dir / "predictions.csv"
    results_submission.write_bytes(submission_path.read_bytes())
    final_timing.update(
        {
            "candidate": asdict(best_candidate),
            "candidate_id": best_candidate.candidate_id,
            "prediction_values": sorted(np.unique(final_predictions).tolist()),
            "submission_path": submission_path,
            "has_header": False,
            "has_index": False,
            "input_sha256": checksums,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output_dir / "final_fit.json", final_timing)
    print(
        f"Wrote {len(final_predictions)} predictions to {submission_path}; "
        f"full-data vectorizer={final_timing['vectorizer_fit_seconds']:.2f}s, "
        f"fit={final_timing['estimator_fit_seconds']:.2f}s, "
        f"predict={final_timing['prediction_seconds']:.2f}s, "
        f"features={final_timing['num_features']}, "
        f"warnings={len(final_timing['warnings'])}"
    )

    config.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_candidate_id": best_candidate.candidate_id,
            "best_validation_accuracy": best_metrics["accuracy"],
            "best_validation_macro_f1": best_metrics["macro_f1"],
            "plot_generation": plot_status,
            "final_fit_warnings": final_timing["warnings"],
        }
    )
    write_json(output_dir / "run_config.json", config)


class Tee:
    """Mirror writes to the terminal and an on-disk run log."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train_data.csv and test_data_unlabeled.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for metrics, manifests, logs and plots",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path(__file__).resolve().parent / "predictions.csv",
        help="Final prediction path (written without header or index)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Fixed stratified split and estimator seed (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced grid for a smoke check; default runs the full 60 settings",
    )
    parser.add_argument(
        "--stability-only",
        action="store_true",
        help="Evaluate the existing best_config.json over stratified splits without retuning",
    )
    parser.add_argument(
        "--stability-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for --stability-only (default: 13 42 73 101 137)",
    )
    return parser.parse_args()


def record_failed_run(args: argparse.Namespace, output_dir: Path, exc: Exception) -> None:
    """Mark a failed main/stability invocation so old result files are not ambiguous."""

    stability_only = bool(args.stability_only)
    manifest_path = output_dir / (
        "stability_run_config.json" if stability_only else "run_config.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    manifest.update(
        {
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "command_argv": [sys.executable, *sys.argv],
            "command_line": shlex.join([sys.executable, *sys.argv]),
            "python_executable": sys.executable,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    try:
        manifest["input_sha256_at_failure"] = input_checksums(args.data_dir.resolve())
    except OSError as checksum_error:
        manifest["input_checksum_error"] = str(checksum_error)
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / ("stability.log" if args.stability_only else "run.log")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            sys.stdout = Tee(original_stdout, log_handle)  # type: ignore[assignment]
            sys.stderr = Tee(original_stderr, log_handle)  # type: ignore[assignment]
            try:
                load_dependencies()
                if args.stability_only:
                    seeds = tuple(args.stability_seeds or DEFAULT_STABILITY_SEEDS)
                    if len(set(seeds)) != len(seeds):
                        raise ValueError("Stability seeds must not contain duplicates")
                    print("=== Locked-configuration split sensitivity assessment ===")
                    print(f"Seeds: {list(seeds)}")
                    run_stability_assessment(args.data_dir.resolve(), output_dir, seeds)
                    print(f"Wrote stability_results.csv and stability_summary.json in {output_dir}")
                else:
                    if args.stability_seeds is not None:
                        raise ValueError("Use --stability-seeds together with --stability-only")
                    run_full_experiment(args, output_dir)
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                try:
                    record_failed_run(args, output_dir, exc)
                except Exception as manifest_error:
                    print(
                        f"Could not write failure manifest: {manifest_error}",
                        file=sys.stderr,
                    )
                return 1
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
