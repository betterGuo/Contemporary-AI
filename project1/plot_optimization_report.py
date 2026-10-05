#!/usr/bin/env python3
"""Render two report-focused figures from completed optimization artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results_optimized")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    results = json.loads((results_dir / "optimization_results.json").read_text(encoding="utf-8"))
    if results.get("status") != "completed":
        raise RuntimeError("Optimization results must have status=completed before plotting")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selection = results["selection"]
    rows = results["candidate_results"]
    score_rows: dict[str, dict[str, Any]] = {}
    baseline = next(row for row in rows if row.get("stage") == "baseline_reference"
                    and row.get("arm") == "baseline" and row.get("seed") == 42)
    score_rows["Baseline\nword 20k · min_df=1 · C=1"] = baseline
    stage2_order = ["word_only", "char_only", "fusion_w1_c025", "fusion_w1_c05", "fusion_w1_c1", "fusion_w05_c1"]
    for arm in stage2_order:
        row = next(row for row in rows if row.get("stage") == "stage2" and row.get("arm") == arm
                   and row.get("seed") == 42 and row.get("status") == "completed")
        labels = {
            "word_only": "Word only\n80k · min_df=1 · C=.5",
            "char_only": "Char only\nchar_wb 80k · min_df=2 · C=.5",
            "fusion_w1_c025": "Fusion · word:char 1:.25\n80k · C=.5",
            "fusion_w1_c05": "Fusion · word:char 1:.5\n80k · C=.5",
            "fusion_w1_c1": "Fusion · word:char 1:1\n80k · C=.5",
            "fusion_w05_c1": "Fusion · word:char .5:1\n80k · C=.5",
        }
        score_rows[labels[arm]] = row
    labels = list(score_rows)
    values = [float(score_rows[label]["validation_metrics"]["macro_f1"]) for label in labels]
    colors = ["#777777", "#287c78", "#c38b35", "#4679a5", "#4679a5", "#1f5a8a", "#4679a5"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    positions = np.arange(len(labels))
    for y, value, color in zip(positions, values, colors):
        ax.plot(value, y, marker="o", markersize=8, color=color, linestyle="none")
        ax.text(value + 0.00018, y, f"{value:.4f}", va="center", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.930, 0.951)
    ax.set_xticks([0.930, 0.935, 0.940, 0.945, 0.950])
    ax.set_xlabel("Seed-42 validation Macro-F1 (axis starts at 0.930; labels show full values)")
    ax.set_title("Ablation: word 80k/min_df=1; char_wb 80k/min_df=2; best arms C=.5 (baseline C=1)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "report_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    predictions_path = results_dir / "validation_predictions.csv"
    import csv
    with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        prediction_rows = list(csv.DictReader(stream))
    winner = selection["final_winner_locked_before_stage3"]
    winner_rows = [row for row in prediction_rows
                   if row["candidate_id"] == winner and row["seed"] == "42"]
    # Several stage aliases may point to the same fitted candidate; deduplicate by original row.
    unique: dict[int, tuple[int, int]] = {}
    for row in winner_rows:
        unique[int(row["original_index"])] = (int(row["target"]), int(row["prediction"]))
    if not unique:
        raise RuntimeError(f"No seed-42 validation prediction rows found for locked winner {winner}")
    confusion = np.zeros((10, 10), dtype=int)
    for true_label, predicted in unique.values():
        confusion[true_label, predicted] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label ID"); ax.set_ylabel("True label ID")
    ax.set_title(f"Seed-42 validation confusion matrix — locked winner {winner}")
    threshold = float(confusion.max()) / 2.0
    for true_label in range(10):
        for predicted in range(10):
            value = int(confusion[true_label, predicted])
            ax.text(predicted, true_label, str(value), ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Validation examples")
    fig.tight_layout()
    fig.savefig(results_dir / "report_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    status = {"generated": ["report_ablation.png", "report_confusion_matrix.png"],
              "winner": winner, "validation_rows": len(unique),
              "note": "Figures are generated from saved validation artifacts; no model was fitted."}
    (results_dir / "report_plot_generation.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
