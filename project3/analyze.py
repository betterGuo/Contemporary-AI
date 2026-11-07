"""Summarize a completed 2x2 CIFAR-10 experiment from saved CPU-side arrays.

Usage: python analyze.py [--results results]. Accuracy fields in input use [0, 1].
No training or network access. Writes the derived artifacts to results/analysis/.
"""
import argparse
import csv
from itertools import combinations
import os
from pathlib import Path
import tempfile

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cifar10-mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ("baseline", "augment", "attention", "both")
NAMES = ("Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck")
ZH = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车")
DISPLAY = dict(zip(RUNS, ("Baseline", "Augmentation", "Attention", "Aug. + attention")))
COLOR = dict(zip(RUNS, ("#345f94", "#008979", "#bd6318", "#84518d")))
FIELDS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")


def load_inputs(root):
    images = np.load(root / "test_images.npy", mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.shape[1:] != (32, 32, 3):
        raise ValueError("Expected uint8 CIFAR images in NHWC order.")
    H, P, C = {}, {}, {}
    y = epochs = None
    for key in RUNS:
        with (root / key / "history.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"Empty history: {key}")
        h = {k: np.array([float(r[k]) for r in rows]) for k in FIELDS}
        if (not all(np.isfinite(v).all() for v in h.values()) or
                any((h[k] < 0).any() for k in FIELDS) or
                any((h[k] > 1).any() for k in ("train_acc", "val_acc")) or
                not np.all(h["epoch"] == h["epoch"].astype(int)) or
                not np.all(np.diff(h["epoch"]) == 1)):
            raise ValueError(f"Invalid epochs, losses, or accuracy in {key}.")
        if epochs is not None and not np.array_equal(h["epoch"], epochs):
            raise ValueError("Epoch budget differs across groups.")
        H[key], epochs = h, h["epoch"]
        with np.load(root / key / "predictions.npz", allow_pickle=False) as data:
            truth, pred, conf = (data[f] for f in ("y_true", "y_pred", "confidence"))
        for v in (truth, pred):
            if (v.shape != (len(images),) or not np.issubdtype(v.dtype, np.integer)
                    or not ((v >= 0) & (v < 10)).all()):
                raise ValueError(f"Invalid test labels/predictions in {key}.")
        if (conf.shape != truth.shape or not np.isfinite(conf).all()
                or not ((conf >= 0) & (conf <= 1)).all()):
            raise ValueError(f"Invalid prediction confidence in {key}.")
        if y is not None and not np.array_equal(y, truth):
            raise ValueError("Test label sequences differ; paired analysis is invalid.")
        y, P[key], C[key] = truth.astype(int), pred.astype(int), conf
    if (np.bincount(y, minlength=10) == 0).any():
        raise ValueError("All ten classes must have at least one example.")
    return images, H, P, C, y


def csvout(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


def savefig(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_history(out, H):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True, sharey="row", layout="constrained")
    for col, k in enumerate(RUNS):
        h = H[k]
        for row, metric in enumerate(("loss", "acc")):
            ax = axes[row, col]
            for split, style, color in (("train", "-", "#345f94"), ("val", "--", "#c05c1b")):
                ax.plot(h["epoch"], h[f"{split}_{metric}"] * (100 if row else 1), style,
                        label=split.capitalize(), color=color, lw=2)
            ax.grid(alpha=.2)
            if not col:
                ax.set_ylabel("Accuracy (%)" if row else "Cross-entropy loss")
            if row:
                ax.set_xlabel("Epoch")
                ax.set_ylim(0, 101)
            else:
                ax.set_title(DISPLAY[k])
        axes[0, col].legend(frameon=False, fontsize=8)
    fig.suptitle("Learning curves | identical epoch budget", fontsize=16)
    savefig(fig, out / "learning_curves.png")


def draw_cm(out, mats):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    for ax, k in zip(axes.flat, RUNS):
        m = mats[k]
        rates = 100 * m / m.sum(1, keepdims=True)
        im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for r in range(10):
            for c in range(10):
                ax.text(c, r, str(m[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if rates[r, c] > 55 else "#18324b")
        ax.set(xticks=range(10), yticks=range(10), yticklabels=NAMES, xlabel="Predicted", ylabel="True")
        ax.set_xticklabels(NAMES, rotation=40, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(DISPLAY[k])
    fig.colorbar(im, ax=axes, shrink=.7, label="Share within true class (%)")
    fig.suptitle("Test confusion matrices | text = sample count", fontsize=16)
    savefig(fig, out / "confusion_matrices.png")


def effects(a, u, t, b):
    """Conditional, average marginal, and difference-in-differences contrasts, in pp."""
    return {
        "augment_without_attention": u-a, "augment_with_attention": b-t,
        "attention_without_augment": t-a, "attention_with_augment": b-u,
        "mean_augment_effect": ((u-a)+(b-t))/2,
        "mean_attention_effect": ((t-a)+(b-u))/2,
        "interaction": (b-u)-(t-a), "both_minus_baseline": b-a,
    }


def paired_intervals(y, predictions, draws=2000, seed=20261107):
    """Stratified paired bootstrap using counts of the 16 possible correctness patterns.

    Resampling pattern counts is exactly equivalent in distribution to resampling
    test images with replacement within each true class. All four models see the
    same resampled images. This method does not resample runs or training seeds.
    """
    rng = np.random.default_rng(seed)
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
    patterns = sum((predictions[k] == y).astype(int) << i for i,k in enumerate(RUNS))
    accumulated = np.zeros((draws,4))
    intervals = {}
    def bounds(samples):
        return {k:np.percentile(v,[2.5,97.5]) for k,v in effects(*samples.T).items()}
    for c in range(10):
        counts = np.bincount(patterns[y == c], minlength=16)
        n = int(counts.sum())
        sampled = rng.multinomial(n, counts/n, size=draws) @ bits
        intervals[c] = bounds(100 * sampled / n)
        accumulated += sampled
    intervals[-1] = bounds(100 * accumulated / len(y))
    return intervals


def example_grid(out, key, images, truth, pred, confidence):
    """One reproducibly sampled misclassification per true class, not top-scored examples."""
    fig, axes = plt.subplots(2, 5, figsize=(11, 5.3), layout="constrained")
    records = []
    for c, ax in enumerate(axes.flat):
        ids = np.flatnonzero((truth == c) & (pred != c))
        ax.axis("off")
        if not ids.size:
            ax.set_title(f"{NAMES[c]}: no error", fontsize=9)
            continue
        i = int(np.random.default_rng(7018+c).choice(ids))
        ax.imshow(images[i], interpolation="nearest")
        ax.set_title(f"#{i}: {NAMES[c]} → {NAMES[pred[i]]}\npred. score {confidence[i]:.2f}", fontsize=9)
        records.append([key, i, NAMES[c], NAMES[pred[i]], float(confidence[i])])
    fig.suptitle(f"{DISPLAY[key]} | one random error per true class (fixed seed)", fontsize=14)
    savefig(fig, out / f"errors_{key}.png")
    return records


def compare_examples(out, images, y, P, C):
    base, both = P["baseline"] == y, P["both"] == y
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    rows = []
    for r, (label, mask) in enumerate((("Corrected", ~base & both), ("Regressed", base & ~both))):
        ids = np.flatnonzero(mask)
        ids = np.random.default_rng(7018+r).choice(ids, size=min(4,len(ids)), replace=False)
        for col, ax in enumerate(axes[r]):
            ax.axis("off")
            if col >= len(ids):
                continue
            i = int(ids[col])
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(f"{label} #{i} · true {NAMES[y[i]]}\n"
                         f"{NAMES[P['baseline'][i]]} → {NAMES[P['both'][i]]}", fontsize=9)
            rows.append([label, i, NAMES[y[i]], *[NAMES[P[k][i]] for k in RUNS],
                         *[float(C[k][i]) for k in RUNS]])
    fig.suptitle("Same test images | baseline prediction → combined prediction", fontsize=14)
    savefig(fig, out / "paired_examples.png")
    csvout(out / "paired_examples.csv", ["type", "test_index", "true_class",
           *[k+"_pred" for k in RUNS], *[k+"_confidence" for k in RUNS]], rows)


def analyze(root):
    root = Path(root)
    images, H, P, C, y = load_inputs(root)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    mats, scores, rc = {}, {}, {}
    classes, runs, pairs, selected, error_pairs, contrasts = [], [], [], [], [], []
    for k in RUNS:
        m = np.bincount(y * 10 + P[k], minlength=100).reshape(10, 10)
        mats[k] = m
        correct = m.diagonal()
        rc[k] = 100 * correct / m.sum(1)
        scores[k] = 100 * correct.sum() / len(y)
        h = H[k]
        peak = int(np.argmax(h["val_acc"]))
        runs.append([k,len(y),int(correct.sum()),scores[k],len(h["epoch"]),
                     int(h["epoch"][peak]),100*h["val_acc"][peak]])
        for c in range(10):
            classes.append([k, NAMES[c], m[c].sum(), correct[c], rc[k][c]])
        for r in range(10):
            for c in range(10):
                if r != c:
                    error_pairs.append([k,NAMES[r],NAMES[c],m[r,c],100*m[r,c]/m[r].sum()])
        csvout(out / f"confusion_{k}.csv", ["true/predicted", *NAMES],
               ([NAMES[c], *row] for c, row in enumerate(m)))
        selected.extend(example_grid(out, k, images, y, P[k], C[k]))
    for a,b in combinations(RUNS,2):
        x,z = P[a]==y, P[b]==y
        n01,n10 = int((~x & z).sum()), int((x & ~z).sum())
        pairs.append([a,b,int((x & z).sum()),n01,n10,int((~x & ~z).sum()),100*(n01-n10)/len(y)])
    intervals = paired_intervals(y,P)
    for c in [-1,*range(10)]:
        e = effects(*[scores[k] if c < 0 else rc[k][c] for k in RUNS])
        contrasts.extend(["overall" if c<0 else NAMES[c],name,value,*intervals[c][name]] for name,value in e.items())
    csvout(out/"summary.csv",["run","n_test","n_correct","test_accuracy_pct","epochs",
                               "first_peak_val_epoch","peak_val_accuracy_pct"],runs)
    csvout(out/"class_accuracy.csv",["run","class","support","correct","accuracy_pct"],classes)
    csvout(out/"pairwise.csv",["from","to","both_correct","corrected","regressed","both_wrong","delta_pp"],pairs)
    csvout(out/"factorial_effects.csv",["class","contrast","delta_pp","bootstrap_p025_pp","bootstrap_p975_pp"],contrasts)
    csvout(out/"error_samples.csv",["run","test_index","true_class","predicted_class","confidence"],selected)
    csvout(out/"directed_errors.csv",["run","true_class","predicted_class","count","true_class_pct"],error_pairs)
    csvout(out/"paired_predictions.csv",["test_index","true_class",*[k+"_prediction" for k in RUNS],
                                         *[k+"_confidence" for k in RUNS]],
           ([i,NAMES[c],*[NAMES[P[k][i]] for k in RUNS],*[float(C[k][i]) for k in RUNS]] for i,c in enumerate(y)))
    draw_history(out,H)
    draw_cm(out,mats)
    compare_examples(out,images,y,P,C)
    fig, ax = plt.subplots(figsize=(11,4.5), layout="constrained")
    for i,k in enumerate(RUNS):
        ax.bar(np.arange(10)+(i-1.5)*.2,rc[k],width=.19,label=DISPLAY[k],color=COLOR[k])
    ax.set(xticks=range(10),xticklabels=NAMES,ylabel="Accuracy within class (%)",ylim=(0,106))
    ax.legend(frameon=False,ncol=4)
    ax.set_title("Which classes benefit? | same test set, four runs")
    savefig(fig,out/"class_accuracy.png")
    e = effects(*(scores[k] for k in RUNS))
    lines = ["# CIFAR-10 四组对照实验", "",
             f"四组使用同一批 {len(y):,} 张测试图像，均训练 {len(H['baseline']['epoch'])} 轮。准确率为预测正确数除以样本数；表中“验证峰值轮”是首次取得最高验证准确率的轮次。", "",
             "| 组别 | 正确数 | 测试准确率 | 验证峰值 | 验证峰值轮 |",
             "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {n} | {s:.2f}% | {v:.2f}% | {ep} |" for k,_,n,s,_,ep,v in runs]
    lines += ["", "## 两个因素各起了多大作用？", "",
              f"固定注意力为关，加入增强后准确率变化 **{e['augment_without_attention']:+.2f}** 个百分点；"
              f"固定注意力为开，相应变化为 **{e['augment_with_attention']:+.2f}** 个百分点。",
              f"固定增强为关，加入注意力后变化 **{e['attention_without_augment']:+.2f}** 个百分点；"
              f"固定增强为开，相应变化为 **{e['attention_with_augment']:+.2f}** 个百分点。",
              f"将另一因素的两种设置等权平均，增强和注意力的平均效应分别为 {e['mean_augment_effect']:+.3f} 和 {e['mean_attention_effect']:+.3f} 个百分点。交互项（两种增强设置下注意力增益之差）为 {e['interaction']:+.2f} 个百分点；它描述本次四组实测的非加性，单次训练无法估计不同随机种子的波动。", "",
              "## 同一张图上发生了什么变化？", ""]
    for a,b in (("baseline","attention"),("augment","both")):
        pair = next(r for r in pairs if r[:2]==[a,b])
        lines.append(f"- {a} → {b}：加入注意力后纠正 {pair[3]} 张，新增错分 {pair[4]} 张。")
    lines.append("")
    row = next(r for r in pairs if r[0]=="baseline" and r[1]=="both")
    lines.append(f"从基线到组合组，{row[3]} 张原本错分的图像被纠正，另有 {row[4]} 张由正确变为错误，两者之差对应 {row[6]:+.2f} 个百分点。另有 {row[5]} 张图像两组都分错，包括预测成不同错误类别的情形。paired_examples.png 从纠正和退化两组中分别按固定种子最多抽取四张。")
    delta = rc["both"]-rc["baseline"]
    cmax,cmin = int(np.argmax(delta)),int(np.argmin(delta))
    lines += [f"各类中，变化最大的是{ZH[cmax]}（{rc['baseline'][cmax]:.2f}% → {rc['both'][cmax]:.2f}%），最小的是{ZH[cmin]}（{rc['baseline'][cmin]:.2f}% → {rc['both'][cmin]:.2f}%）。完整十类结果见 class_accuracy.csv。", "",
              "## 各组最常见的定向错误", ""]
    for k in RUNS:
        m = mats[k].copy()
        np.fill_diagonal(m,0)
        if m.sum():
            a,b = np.unravel_index(m.argmax(),m.shape)
            lines.append(f"- {k}：{ZH[a]} → {ZH[b]}，{m[a,b]} 张，占真实{ZH[a]}样本的 {100*m[a,b]/mats[k][a].sum():.2f}%。")
        else:
            lines.append(f"- {k}：本测试集没有错分样本。")
    lines += ["","误分类图 errors_*.png 按真实类别分别抽样，每类最多一张，编号对应官方测试集顺序。图中分数为模型对预测类别给出的置信度。通过这些个例可以观察混淆对象；它们不能单独说明某种背景或部位导致了误判。","","## 复核入口","",
              "- learning_curves.png：各组训练和验证的损失、准确率。",
              "- confusion_matrices.png / confusion_*.csv：行是真实类别、列是预测类别；图中文字为样本数，颜色按真实类别归一化。",
              "- class_accuracy.png / class_accuracy.csv：逐类正确数、样本数和准确率；summary.csv：总准确率。",
              "- paired_predictions.csv：逐图四组预测类别、预测分数；pairwise.csv：所有组对的纠正与退化。",
              "- factorial_effects.csv：总体及各类的条件效应、平均效应和交互项，单位为百分点。",
              "- directed_errors.csv：所有定向混淆数及其占真实类别的比例。", ""]
    changes = mats["both"] - mats["baseline"]
    src, dst = np.where(~np.eye(10, dtype=bool))
    ranks = np.argsort(changes[src,dst], kind="stable")
    reductions = [int(j) for j in ranks if changes[src[j],dst[j]] < 0][:3]
    increases = [int(j) for j in ranks[::-1] if changes[src[j],dst[j]] > 0][:3]
    i = lines.index("## 各组最常见的定向错误")
    change_lines = ["观察从基线到组合组的定向混淆计数，可以定位净准确率变化来自哪些错误类型："]
    for label,idxs in (("减少最多", reductions), ("增加最多", increases)):
        if idxs:
            text = []
            for j in idxs:
                a,b = src[j],dst[j]
                text.append(f"{ZH[a]}→{ZH[b]}（{mats['baseline'][a,b]} → {mats['both'][a,b]} 张）")
            change_lines.append("- "+label+"："+"、".join(text)+"。")
    if not reductions and not increases:
        change_lines.append("- 两组定向混淆的计数完全一致。")
    lines[i:i] = change_lines+[""]
    i = lines.index("## 同一张图上发生了什么变化？")
    ci_text = []
    for field,name in (("mean_augment_effect","增强平均效应"),("mean_attention_effect","注意力平均效应"),("interaction","交互项")):
        lo,hi = intervals[-1][field]
        ci_text.append(f"{name} [{lo:+.2f}, {hi:+.2f}]")
    lines[i:i] = ["对测试图像按真实类别分层，在各层内有放回地抽样 2,000 次，每次四组模型共享抽中的样本。95% 配对百分位区间（百分点）："
                  + "；".join(ci_text) + "。这些区间仅描述固定模型在同类测试样本重抽样时的波动，不涵盖更换训练种子产生的差异。",""]
    (out/"FINDINGS.md").write_text("\n".join(lines),encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path(__file__).resolve().parent/"results")
    print(analyze(p.parse_args().results))
