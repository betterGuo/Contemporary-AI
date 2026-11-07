"""从已保存的真实实验数据生成报告用 PNG / SVG，不重新运行搜索实验。

图内使用英文与 DejaVu Sans，避免不同机器缺少中文字体。
耗时区间来自原始运行样本或同一配置的样本分位数，不平均分位数。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .pathfinding import Grid
from .tasks import load_sample


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
INK = "#243746"
MUTED = "#5B6874"
GRID_COLOR = "#DFE5E9"
PATH_VARIANTS = ("dijkstra", "goal_aware_fifo", "goal_aware_larger_g", "alt_larger_g")
PATH_LABELS = ("Dijkstra", "Goal-aware / FIFO", "Goal-aware / larger-g", "ALT / larger-g")
GAME_CONFIGS = (
    ("minimax", "natural", False, "Minimax"),
    ("minimax", "natural", True, "Minimax + TT"),
    ("alpha_beta", "natural", False, "Alpha-beta / natural"),
    ("alpha_beta", "reverse", False, "Alpha-beta / reverse"),
    ("alpha_beta", "greedy", False, "Alpha-beta / greedy"),
    ("alpha_beta", "greedy", True, "Alpha-beta / greedy + TT"),
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _one(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"绘图需要一条匹配记录，找到 {len(matches)} 条: {criteria}")
    return matches[0]


def _quartiles(values):
    values = list(values)
    if not values:
        raise ValueError("不能对空数据绘图")
    if len(values) == 1:
        return values[0], values[0], values[0]
    q25, _, q75 = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), q25, q75


def _style(ax, *, horizontal=False, log=False):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.grid(axis="x" if horizontal else "y", color=GRID_COLOR, linewidth=0.8)
    if log:
        (ax.set_xscale if horizontal else ax.set_yscale)("log")


def _save(fig, directory: Path, stem: str) -> list[Path]:
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _number_labels(ax, bars, *, horizontal=False, color=INK, fmt="{:.0f}"):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if horizontal:
            ax.annotate(fmt.format(value), (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                        color=color, fontsize=11)
        else:
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 5 if value >= 0 else -5), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top",
                        color=color, fontsize=11)


def _maps(samples, directory):
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), layout="constrained")
    for problem, ax in enumerate(axes, start=1):
        source = load_sample(problem)
        grid = Grid.from_rows(source["grid"], marker_costs=source.get("marker_costs"))
        result = samples[f"problem{problem}"]
        path = result["path"]
        start, goal = tuple(source["start"]), tuple(source["goal"])
        maximum = max(value for row in grid.costs for value in row if value is not None)
        for row in range(grid.rows):
            for col in range(grid.cols):
                value = grid.costs[row][col]
                if value is None:
                    ax.add_patch(Rectangle((col - .5, row - .5), 1, 1,
                                           facecolor="#34454F", edgecolor="white", linewidth=2))
                    ax.text(col, row, "X", color="white", ha="center", va="center", fontsize=19)
                    continue
                # 正权重越高蓝色越深；路径本身采用对比鲜明的橙色。
                shade = plt.get_cmap("Blues")(.04 + .22 * value / max(1, maximum))
                ax.add_patch(Rectangle((col - .5, row - .5), 1, 1,
                                       facecolor=shade, edgecolor="#CDD9DF", linewidth=.9))
                ax.text(col - .36, row - .31, str(value), color=MUTED, ha="left", va="center", fontsize=12)
        columns, rows = [pos[1] for pos in path], [pos[0] for pos in path]
        ax.plot(columns, rows, color=COLORS[1], linewidth=4.5, marker="o",
                markersize=6.5, markeredgecolor="white", markeredgewidth=1, zorder=4)
        for position, label, color in ((start, "S", COLORS[0]), (goal, "E", COLORS[2])):
            row, col = position
            ax.scatter([col], [row], s=650, facecolor=color, edgecolor="white", linewidth=2, zorder=5)
            ax.text(col, row, label, ha="center", va="center", color="white", weight="bold", fontsize=17, zorder=6)
        steps = len(path) - 1
        total = samples["answers"]["problem1_steps" if problem == 1 else "problem2_cost"]
        ax.set_title(f"Problem {problem}: {steps} moves, total cost {total}", loc="left", pad=17)
        ax.set_xticks(range(grid.cols))
        ax.set_yticks(range(grid.rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xlim(-.5, grid.cols - .5)
        ax.set_ylim(grid.rows - .5, -.5)
        ax.set_aspect("equal")
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Verified shortest paths on the supplied grids", fontsize=20, weight="bold")
    fig.supxlabel("Cell numbers = entry costs; the starting cell is free.  Problem 2 enters E at cost 0.",
                  fontsize=12, color=MUTED)
    return _save(fig, directory, "01_required_paths")


def _required_astar(summary, directory):
    modes = (("zero", "fifo"), ("manhattan", "fifo"), ("goal_aware", "fifo"), ("goal_aware", "larger_g"))
    labels = ["Dijkstra (h = 0)", "Raw Manhattan", "Goal-aware / FIFO", "Goal-aware / larger-g"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True, layout="constrained")
    for problem, ax in enumerate(axes, start=1):
        rows = [_one(summary, suite="required", case=f"problem{problem}", heuristic=h, tie_break=t)
                for h, t in modes]
        values = [row["expanded"] for row in rows]
        bars = ax.barh(range(4), values, color=COLORS[:4], height=.6)
        _number_labels(ax, bars, horizontal=True)
        if problem == 2:
            bars[1].set_hatch("///")
            bars[1].set_edgecolor("#775200")
        ax.set_yticks(range(4), labels)
        ax.set_xlim(0, max(values) * 1.23)
        ax.set_title(f"Problem {problem}", loc="left", pad=15)
        ax.set_xlabel("Expanded nodes (goal excluded)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        _style(ax, horizontal=True)
    axes[0].invert_yaxis()
    fig.suptitle("A* heuristic and tie-breaking comparison", fontsize=20, weight="bold")
    fig.supxlabel("Raw Manhattan is inadmissible with a zero-cost goal; the default uses a consistent goal-aware lower bound.",
                  color="#A45A00", fontsize=12)
    return _save(fig, directory, "02_required_astar")


def _tree(summary, directory):
    configurations = [(a, o) for a in ("minimax", "alpha_beta") for o in ("natural", "reverse")]
    rows = [_one(summary, suite="required", case="problem3", algorithm=a, order=o)
            for a, o in configurations]
    labels = [f"{'Minimax' if a == 'minimax' else 'Alpha-beta'}\n{o}" for a, o in configurations]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), layout="constrained")
    for ax, fields, names, title in (
        (axes[0], ("visited", "leaves"), ("Recursive visits", "Leaf evaluations"), "Search work"),
        (axes[1], ("cutoffs", "pruned_branches"), ("Cutoff events", "Skipped direct branches"), "Recorded pruning"),
    ):
        for i, (field, name) in enumerate(zip(fields, names)):
            bars = ax.bar([x + (i - .5) * .34 for x in range(4)], [row[field] for row in rows],
                          width=.32, label=name, color=COLORS[i])
            _number_labels(ax, bars)
        ax.set_xticks(range(4), labels)
        ax.set_title(title, loc="left", pad=15)
        ax.set_ylabel("Count")
        top = max(row[field] for row in rows for field in fields)
        ax.set_ylim(0, max(1, top) * 1.35)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.legend(loc="upper right", fontsize=11)
        _style(ax)
    fig.suptitle("Problem 3: the same root value, different traversal work", fontsize=20, weight="bold")
    fig.supxlabel("All four searches return 5. Skipped branches count direct siblings, not every node in their subtrees.",
                  fontsize=12, color=MUTED)
    return _save(fig, directory, "03_game_tree_pruning")


def _required_treasures(summary, samples, directory):
    configurations = [("minimax", "natural"), *(('alpha_beta', order) for order in ("natural", "reverse", "greedy"))]
    names = ["Minimax baseline", "Alpha-beta / natural", "Alpha-beta / reverse", "Alpha-beta / greedy"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    for index, depth in enumerate((2, 4)):
        rows = [_one(summary, suite="required", case="problem4", depth=depth, algorithm=a, order=o, cache=False)
                for a, o in configurations]
        bars = axes[0].barh([y + (index - .5) * .34 for y in range(4)], [row["visited"] for row in rows],
                            height=.31, color=COLORS[index], label=f"H = {depth}")
        _number_labels(axes[0], bars, horizontal=True)
    axes[0].set_yticks(range(4), names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Recursive visits")
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 1.17)
    axes[0].set_title("Traversal order (cache disabled)", loc="left", pad=15)
    axes[0].legend(loc="lower right")
    _style(axes[0], horizontal=True)

    root_values = samples["problem4"]["exact_root_action_values"]
    all_values = []
    for index, depth in enumerate((2, 4)):
        values = [root_values[str(depth)][f"T{i}"] for i in range(1, 5)]
        all_values.extend(values)
        bars = axes[1].bar([x + (index - .5) * .34 for x in range(4)], values, width=.32,
                           label=f"H = {depth}", color=COLORS[index])
        _number_labels(axes[1], bars)
        for bar, value in zip(bars, values):
            if value == max(values):
                bar.set_edgecolor(INK)
                bar.set_linewidth(2.2)
    axes[1].set_xticks(range(4), [f"T{i}" for i in range(1, 5)])
    axes[1].set_ylabel("Exact root-action utility (A - B)")
    axes[1].set_title("Search depth changes the best first move", loc="left", pad=15)
    axes[1].axhline(0, color=INK, linewidth=.8)
    axes[1].set_ylim(min(all_values) - 2, max(all_values) + 2)
    axes[1].legend(loc="upper right")
    _style(axes[1])
    fig.suptitle("Problem 4: horizon and move ordering", fontsize=20, weight="bold")
    fig.supxlabel("Root-action values come from independent full-window Minimax searches; outlined bars mark each depth's optimum.",
                  fontsize=12, color=MUTED)
    return _save(fig, directory, "04_treasure_horizon_ordering")


def _landmarks(summary, amortization, manifest, directory):
    path_rows = [row for row in summary if row["suite"] == "path_extension"]
    cases = list(dict.fromkeys(row["case"] for row in path_rows))
    base_seed = manifest["seed"]
    labels = []
    for case in cases:
        row = next(row for row in path_rows if row["case"] == case)
        kind = {"open": "Open", "walls": "Walls", "weighted": "Weighted", "zero_cost": "Zero-cost"}[row["kind"]]
        labels.append(f"{kind}\nseed +{row['seed'] - base_seed}")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.6), layout="constrained")
    width = .19
    for index, (variant, label) in enumerate(zip(PATH_VARIANTS, PATH_LABELS)):
        positions = [x + (index - 1.5) * width for x in range(len(cases))]
        expanded = [sum(row["expanded"] for row in path_rows if row["case"] == case and row["variant"] == variant)
                    for case in cases]
        axes[0].bar(positions, expanded, width=width * .91, color=COLORS[index], label=label)
        medians, lower, upper, online = [], [], [], []
        for case in cases:
            runs = [row for row in amortization if row["case"] == case and row["variant"] == variant]
            median, q25, q75 = _quartiles(float(row["total_ns"]) / 1e6 for row in runs)
            medians.append(median)
            lower.append(median - q25)
            upper.append(q75 - median)
            online.append(statistics.median(float(row["online_ns"]) / 1e6 for row in runs))
        axes[1].bar(positions, medians, width=width * .91, color=COLORS[index],
                    yerr=[lower, upper], capsize=2, error_kw={"elinewidth": .9, "ecolor": INK})
        if variant == "alt_larger_g":
            axes[1].scatter(positions, online, marker="D", s=29, facecolor="white", edgecolor=INK,
                            linewidth=1, zorder=5, label="ALT online only (median)")
    for ax in axes:
        ax.set_xticks(range(len(cases)), labels, fontsize=11)
        _style(ax, log=True)
        ax.margins(x=.025)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.7)
    axes[0].set_title("12-query expansion totals", loc="left", pad=15)
    axes[0].set_ylabel("Expanded nodes (log scale)")
    axes[0].legend(loc="upper left", ncol=2, fontsize=10.5)
    axes[1].set_title("Total cost includes one ALT preprocessing pass", loc="left", pad=15)
    axes[1].set_ylabel("Preprocessing + 12 online queries (ms, log scale)")
    axes[1].legend(loc="upper left", fontsize=10.5)
    fig.suptitle("ALT: fewer online expansions must be weighed against preprocessing", fontsize=20, weight="bold")
    fig.supxlabel(f"Base seed = {base_seed}. Times show median / IQR of per-trial component sums; diamonds exclude preprocessing.",
                  fontsize=12, color=MUTED)
    return _save(fig, directory, "05_landmark_cost_tradeoff")


def _game_scaling(summary, raw_runs, manifest, directory):
    rows = [row for row in summary if row["suite"] == "game_extension"]
    cases = list(dict.fromkeys(row["case"] for row in rows))
    base_seed = manifest["seed"]
    case_rows = [next(row for row in rows if row["case"] == case) for case in cases]
    labels = [f"{row['treasures']} treasures\nseed +{row['seed'] - base_seed}" for row in case_rows]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.2), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    hits = []
    for index, (algorithm, order, cache, label) in enumerate(GAME_CONFIGS):
        selected = [_one(rows, case=case, algorithm=algorithm, order=order, cache=cache) for case in cases]
        axes[0].plot(range(len(cases)), [row["visited"] for row in selected], color=COLORS[index],
                     marker=("o", "s", "^", "v", "D", "P")[index], linewidth=1.9, markersize=7,
                     linestyle="--" if cache else "-", label=label)
        # 合并原始运行样本后重新计算中位数和四分位数，绝不平均各 seed 的分位数。
        timings = [float(row["wall_ns"]) / 1e6 for row in raw_runs
                   if row["suite"] == "game_extension" and int(row["treasures"]) == 8
                   and row["algorithm"] == algorithm and row["order"] == order
                   and row["cache"].lower() == str(cache).lower()]
        median, q25, q75 = _quartiles(timings)
        axes[1].errorbar([median], [index], xerr=[[median - q25], [q75 - median]],
                         fmt="o", markersize=8, capsize=5, elinewidth=2.2, color=COLORS[index])
        axes[1].annotate(f"{median:.2f}", (q75, index), xytext=(7, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=11, color=INK)
        if cache:
            median_hits = statistics.median(row["cache_hits"] for row in selected if row["treasures"] == 8)
            hits.append(f"{label}: {median_hits:g}")
    axes[0].set_xticks(range(len(cases)), labels, fontsize=11)
    axes[0].set_title("Visits at full search depth, per instance", loc="left", pad=15)
    axes[0].set_ylabel("Recursive visits (log scale)")
    _style(axes[0], log=True)
    axes[0].legend(loc="upper left", fontsize=10.1, ncol=2)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 5)
    axes[1].set_yticks(range(len(GAME_CONFIGS)), [config[3] for config in GAME_CONFIGS])
    axes[1].invert_yaxis()
    axes[1].set_ylim(len(GAME_CONFIGS) - .5, -.6)
    axes[1].set_title("Eight treasures: pooled median and IQR", loc="left", pad=15)
    axes[1].set_xlabel("Time per search (ms, log scale)")
    _style(axes[1], horizontal=True, log=True)
    axes[1].margins(x=.25)
    seeds = len({row["seed"] for row in rows if row["treasures"] == 8})
    fig.suptitle("Move ordering and transposition tables as the game grows", fontsize=20, weight="bold")
    fig.supxlabel(f"Time distribution pools {seeds} seeds × {manifest['repeats']} runs each. TT = transposition table.\n"
                  + "Median TT hits across eight-treasure instances — " + "; ".join(hits),
                  fontsize=11.5, color=MUTED)
    return _save(fig, directory, "06_game_scaling_cache")


def _reuse_batches(reuse, manifest, directory):
    """连续批次墙钟测量：每个 ALT trial 只建一次索引，并包含在 wall_ns 内。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained", sharey=True)
    upper = 0
    for ax, kind in zip(axes, ("walls", "weighted", "zero_cost")):
        cases = [row for row in reuse if row["kind"] == kind]
        counts = sorted({int(row["query_count"]) for row in cases})
        if not counts:
            raise ValueError(f"reuse_batches.csv 中缺少 {kind} 场景")
        for variant, label, color, marker in (
            ("goal_aware_larger_g", "Goal-aware / larger-g", COLORS[2], "o"),
            ("alt_larger_g", "ALT / larger-g", COLORS[3], "s"),
        ):
            medians, q25s, q75s = [], [], []
            for queries in counts:
                values = [float(row["wall_ns"]) / 1e6 for row in cases
                          if row["variant"] == variant and int(row["query_count"]) == queries]
                median, q25, q75 = _quartiles(values)
                medians.append(median)
                q25s.append(q25)
                q75s.append(q75)
            upper = max(upper, max(q75s))
            ax.plot(counts, medians, marker=marker, color=color, linewidth=2.2, markersize=7, label=label)
            ax.fill_between(counts, q25s, q75s, color=color, alpha=.18, linewidth=0)
            ax.errorbar(counts, medians,
                        yerr=[[m - q for m, q in zip(medians, q25s)], [q - m for m, q in zip(medians, q75s)]],
                        fmt="none", color=color, capsize=3, elinewidth=1)
        ax.set_title({"walls": "Walls", "weighted": "Weighted", "zero_cost": "Zero-cost"}[kind],
                     loc="left", pad=15)
        ax.set_xticks(counts)
        ax.set_xlabel("Queries sharing one index")
        ax.set_xlim(-3, max(counts) * 1.05)
        _style(ax)
    axes[0].set_ylim(0, upper * 1.12)
    axes[0].set_ylabel("Total batch wall time (ms)")
    axes[0].legend(loc="upper left", fontsize=11)
    fig.suptitle("When does ALT preprocessing pay off?", fontsize=20, weight="bold")
    fig.supxlabel(f"Seed = {manifest['seed']}. Every ALT batch includes a fresh index build. Lines = median; shaded bands = IQR.\n"
                  "Measured continuously per batch; query sequences are fixed and reused across the two methods.",
                  fontsize=12, color=MUTED)
    return _save(fig, directory, "07_landmark_reuse_batches")


def make_plots(output: Path) -> list[Path]:
    """读取课例与基准结果，写出六组 PNG / SVG；有复用实验时再生成第七组。

    必需输入：samples.json、summary.json、manifest.json、raw_runs.csv、
    landmark_amortization.csv。图表只消费已有数据，不改变计时结果。
    """
    output = Path(output)
    samples = _read_json(output / "samples.json")
    summary = _read_json(output / "summary.json")
    manifest = _read_json(output / "manifest.json")
    raw_runs = _read_csv(output / "raw_runs.csv")
    amortization = _read_csv(output / "landmark_amortization.csv")
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 15, "axes.titleweight": "semibold",
        "axes.labelsize": 13, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "svg.fonttype": "none",
        "figure.constrained_layout.w_pad": .10, "figure.constrained_layout.h_pad": .10,
        "figure.constrained_layout.wspace": .08, "savefig.facecolor": "white",
    }):
        paths.extend(_maps(samples, directory))
        paths.extend(_required_astar(summary, directory))
        paths.extend(_tree(summary, directory))
        paths.extend(_required_treasures(summary, samples, directory))
        paths.extend(_landmarks(summary, amortization, manifest, directory))
        paths.extend(_game_scaling(summary, raw_runs, manifest, directory))
        if (output / "reuse_batches.csv").exists():
            paths.extend(_reuse_batches(_read_csv(output / "reuse_batches.csv"), manifest, directory))
    return paths
