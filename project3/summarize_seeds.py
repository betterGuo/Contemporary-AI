"""Compare three paired training seeds on one fixed split, using saved predictions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GROUPS = ("baseline", "augment", "attention", "both")
NAMES = dict(zip(GROUPS, ("基线", "仅增强", "仅注意力", "增强＋注意力")))
PAIRS = (
    ("baseline", "augment"), ("attention", "both"),
    ("baseline", "attention"), ("augment", "both"),
    ("baseline", "both"), ("augment", "attention"),
)
FACTORS = {
    "augmentation_main": "增强主效应",
    "attention_main": "注意力主效应",
    "interaction": "交互效应",
}
FLAGS = dict(zip(GROUPS, ((False, False), (True, False), (False, True), (True, True))))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect_sizes(counts, n):
    """Contrasts on the additive test-accuracy (percentage-point) scale."""
    b, a, t, at = [int(counts[g]) for g in GROUPS]
    return dict(
        augmentation_main=((a-b) + (at-t)) * 50 / n,
        attention_main=((t-b) + (at-a)) * 50 / n,
        interaction=((at-a) - (t-b)) * 100 / n,
    )


def switches(a, b):
    """a/b are binary per-image correctness for reference and comparison."""
    return {
        "corrected": int(np.count_nonzero(~a & b)), "regressed": int(np.count_nonzero(a & ~b)),
        "both_correct": int(np.count_nonzero(a & b)), "both_wrong": int(np.count_nonzero(~a & ~b)),
    }


def spread(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sample_sd": float(x.std(ddof=1))}


def directions(values):
    x = np.asarray(values)
    return {"positive": int((x > 0).sum()), "zero": int((x == 0).sum()), "negative": int((x < 0).sum())}


def read_run(folder, group, seed, protocol, split_sha):
    cfg = read_json(folder/"config.json")
    test = read_json(folder/"test.json")
    with (folder/"history.csv").open(encoding="utf-8", newline="") as f:
        h = list(csv.DictReader(f))
    with np.load(folder/"predictions.npz", allow_pickle=False) as p:
        y = p["y_true"]
        pred = p["y_pred"]
    epochs = int(protocol["epochs_per_run"])
    if (cfg["seed"], cfg["group"], cfg["epochs"], cfg["batch"], cfg["split_sha256"]) != (
        seed, group, epochs, protocol["batch_size"], split_sha
    ) or (cfg["augmentation"], cfg["attention"]) != FLAGS[group]:
        raise ValueError(f"{folder}: run differs from preregistered design")
    if [int(row["epoch"]) for row in h] != list(range(1, epochs + 1)):
        raise ValueError(f"{folder}: missing epochs")
    if y.shape != (10000,) or pred.shape != y.shape or not np.array_equal(np.unique(y, return_counts=True)[1], [1000]*10):
        raise ValueError(f"{folder}: incomplete CIFAR-10 test predictions")
    if y.dtype.kind not in "iu" or pred.dtype.kind not in "iu" or not np.isin(y, range(10)).all() or not np.isin(pred, range(10)).all():
        raise ValueError(f"{folder}: invalid test class")
    acc = np.asarray([float(row["val_acc"]) for row in h])
    secs = np.asarray([float(row["seconds"]) for row in h])
    if not np.isfinite(secs).all() or np.any(secs <= 0) or not np.isfinite(acc).all() or not np.all((0 <= acc) & (acc <= 1)):
        raise ValueError(f"{folder}: invalid history")
    best = int(acc.argmax())  # earliest maximum, as in train.py
    ok = y == pred
    if test["best_epoch"] != best+1 or not np.isclose(acc[best], test["val_acc"], rtol=0, atol=1e-7) or not np.isclose(ok.mean(), test["test_acc"], rtol=0, atol=1e-7):
        raise ValueError(f"{folder}: metrics do not match predictions/validation history")
    row = dict(seed=seed, group=group, n_test=len(y), test_correct=int(ok.sum()),
               test_accuracy_pct=100*int(ok.sum())/len(y), best_epoch=best+1,
               best_val_accuracy_pct=acc[best]*100, train_and_val_seconds=secs.sum())
    return y, ok, row, cfg


def plot_results(out, by_seed, factors, seeds):
    os.environ.setdefault("MPLCONFIGDIR", str(out/".mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73")
    for i, seed in enumerate(seeds):
        col = colors[i % len(colors)]
        ax.plot(range(4), [by_seed[seed][g] for g in GROUPS], color=col, alpha=.83, lw=1.5,
                marker=("o", "s", "^")[i % 3], label=str(seed))
        bx.scatter(np.arange(3)+(i-(len(seeds)-1)/2)*.12,
                   [factors[seed][k] for k in FACTORS], color=col, marker=("o","s","^")[i % 3], s=48)
    ax.set_xticks(range(4), ["Baseline", "Aug", "Attn", "Aug + attn"])
    bx.set_xticks(range(3), ["Aug main", "Attention main", "Interaction"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy across the four configurations")
    ax.legend(title="Training seed", fontsize=8)
    bx.set_title("Effects within each paired training seed")
    bx.set_ylabel("Accuracy difference (percentage points)")
    bx.axhline(0, color="#4a4a4a", lw=1, linestyle="--")
    for axis in (ax,bx):
        axis.grid(axis="y", alpha=.25)
        axis.spines[["top","right"]].set_visible(False)
    for ext in ("png","svg"):
        fig.savefig(out/f"seed_comparison.{ext}", dpi=180)
    plt.close(fig)


def summarise(results, protocol_path):
    protocol = read_json(protocol_path)
    seeds = [protocol["primary_training_seed"], *protocol["replication_training_seeds"]]
    if not (len(seeds) >= 2 and len(seeds) == len(set(seeds)) and tuple(protocol["groups"]) == GROUPS):
        raise ValueError("Expected distinct training seeds and four ordered groups")
    ref_y = ref_settings = ref_split = None
    runs, paired, effects = [], [], []
    by_seed, factors = {}, {}
    for ix, seed in enumerate(seeds):
        base = results if ix == 0 else results/"replicates"/str(seed)
        with np.load(base/"split.npz",allow_pickle=False) as sp:
            tr, va = sp["train"], sp["val"]
        split = hashlib.sha256(tr.tobytes()+va.tobytes()).hexdigest()
        if ref_split is not None and split != ref_split:
            raise ValueError("Training/validation split differs across seeds")
        ref_split = split
        ok = {}
        by_seed[seed] = {}
        for group in GROUPS:
            y, ok[group], row, cfg = read_run(base/group,group,seed,protocol,split)
            settings = tuple(cfg[k] for k in ("lr","weight_decay","optimizer","scheduler","model_sha256","device"))
            if ref_settings is None:
                ref_y, ref_settings = y, settings
            if not np.array_equal(y,ref_y) or settings != ref_settings:
                raise ValueError("Test targets/order or shared settings differ")
            runs.append(row)
            by_seed[seed][group] = row["test_accuracy_pct"]
        for a,b in PAIRS:
            counts = switches(ok[a], ok[b])
            paired.append(dict(seed=seed, reference=a, comparison=b, **counts,
                               delta_pp=(counts["corrected"]-counts["regressed"])*100/len(y)))
        factors[seed] = effect_sizes({g:ok[g].sum() for g in GROUPS}, len(y))
        effects.extend(dict(seed=seed, contrast=k, delta_pp=v) for k,v in factors[seed].items())

    out = results/"robustness"
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/"runs.csv",runs)
    save_csv(out/"paired_switches.csv",paired)
    save_csv(out/"factorial_effects.csv",effects)
    stats = []
    for group in GROUPS:
        stats.append(dict(quantity=group, unit="%", **spread([by_seed[s][group] for s in seeds]),
                          positive="", zero="", negative=""))
    for a,b in PAIRS:
        vals = [r["delta_pp"] for r in paired if (r["reference"],r["comparison"]) == (a,b)]
        stats.append(dict(quantity=f"{b} - {a}",unit="pp",**spread(vals),**directions(vals)))
    for key in FACTORS:
        vals = [factors[s][key] for s in seeds]
        stats.append(dict(quantity=key,unit="pp",**spread(vals),**directions(vals)))
    save_csv(out/"summary.csv",stats)
    plot_results(out, by_seed, factors, seeds)
    cols = "| 种子 | 方案 | 测试准确率 | 验证最佳轮 | 最佳验证准确率 | 训练与验证（秒） |"
    lines = ["# 不同训练种子下的增强与注意力对照","",
             f"{len(seeds)} 个训练种子均使用同一训练/验证划分；每个种子内四组训练轮数相同。模型按验证准确率选择，同分取较早轮次。时间是 history.csv 中各轮训练与验证秒数之和，不含模型保存和最终测试。","",
             cols, "|---:|---|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['seed']} | {NAMES[r['group']]} | {r['test_accuracy_pct']:.2f}% | {r['best_epoch']} | {r['best_val_accuracy_pct']:.2f}% | {r['train_and_val_seconds']:.1f} |")
    lines += ["","![各训练种子的实际结果](seed_comparison.png)","","| 方案 | 测试准确率均值 ± 样本标准差（%） |","|---|---:|"]
    for r in stats[:4]:
        lines.append(f"| {NAMES[r['quantity']]} | {r['mean']:.2f} ± {r['sample_sd']:.2f} |")
    lines += ["","比较单位为准确率百分点。增强主效应取“仅增强−基线”和“组合−仅注意力”之均值；注意力主效应同理。交互效应是（组合−仅增强）−（仅注意力−基线），描述加入增强后注意力增益的变化。","",
              "| 对照 | "+" | ".join(map(str,seeds))+" | 均值 ± 样本标准差 | 正 / 零 / 负 |",
              "|---|"+"---:|"*(len(seeds)+2)]
    for k in FACTORS:
        vals = [factors[s][k] for s in seeds]
        r, signs = spread(vals),directions(vals)
        lines.append("| "+FACTORS[k]+" | "+" | ".join(f"{v:+.2f}" for v in vals)+f" | {r['mean']:+.2f} ± {r['sample_sd']:.2f} | {signs['positive']} / {signs['zero']} / {signs['negative']} |")
    lines.append("")
    for k,label in FACTORS.items():
        vals=[factors[s][k] for s in seeds]
        counts=directions(vals)
        if counts["positive"]==len(seeds):
            direction="所有种子均为正"
        elif counts["negative"]==len(seeds):
            direction="所有种子均为负"
        elif counts["zero"]==len(seeds):
            direction="所有种子均为零"
        else:
            direction="各个种子的符号不完全一致"
        lines.append(f"- {label}：{direction}，范围 {min(vals):+.2f} 至 {max(vals):+.2f} 个百分点。")
    lines += ["","下表按相同测试图像配对。纠正指参考组判断错误而比较组判断正确，退化则相反，二者之差对应净新增的正确预测数。全部六组两两比较见 paired_switches.csv。","",
              "| 种子 | 基线 → 组合：纠正 | 退化 | 净增正确样本 |","|---:|---:|---:|---:|"]
    for r in paired:
        if r["reference"]=="baseline" and r["comparison"]=="both":
            lines.append(f"| {r['seed']} | {r['corrected']} | {r['regressed']} | {r['corrected']-r['regressed']:+d} |")
    lines += ["",f"这些均值与标准差描述同一官方测试集、同一数据划分上的 {len(seeds)} 次训练结果。图中的点各代表一个实际训练种子，不能把重复评估当成额外的独立测试图像。"]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--protocol",type=Path,default=ROOT/"protocol.json")
    args=parser.parse_args()
    summarise(args.results,args.protocol)
