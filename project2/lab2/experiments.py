"""课例验证、对照实验和原始数据采集；结果全部来自实际执行。

计时使用 perf_counter_ns；同一组配置按固定种子交错执行，预热不入表。
外层 wall_ns 包含求解入口校验/启发工厂，排除输入读取、序列化和画图。
搜索器内部 elapsed_ns 同时保留，避免混淆计时边界。
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import get_clock_info, perf_counter_ns
from zoneinfo import ZoneInfo
import csv
import json
import math
import platform
import random
import statistics
import sys

from .games import solve_tree, solve_treasure, tree_from_dict
from .pathfinding import Grid, LandmarkIndex, astar
from .tasks import ROOT, load_sample, make_treasure_game


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def path_accumulation(grid, path):
    total, rows = 0, []
    for i, pos in enumerate(path):
        entering = 0 if i == 0 else grid.cost(pos)
        total += entering
        rows.append({"step": i, "position": list(pos), "entering_cost": entering, "cumulative_cost": total})
    return rows


def root_actions(game, depth):
    """独立全窗口求每个首步的精确值，不把剪枝的界误作动作精确值。"""
    state = game.initial_state()
    return {
        target: solve_treasure(game, depth - 1, algorithm="minimax", state=game.successor(state, target)).value
        for target in game.legal_actions(state)
    }


def optimal_line(game, depth):
    """用每个中间状态的全窗口 Minimax 重建最优对局，供过程分析。"""
    state, moves = game.initial_state(), []
    for ply in range(depth):
        result = solve_treasure(game, depth - ply, algorithm="minimax", state=state)
        target = result.best_target
        if target is None:
            break
        start = state.pos_a if state.player == "A" else state.pos_b
        gain = game.action_gain(state, target)
        next_state = game.successor(state, target)
        moves.append({"ply": ply + 1, "player": state.player, "from": start, "target": target,
                      "cost": game.distance(start, target), "gain": gain,
                      "score_a": next_state.score_a, "score_b": next_state.score_b,
                      "utility": next_state.utility, "remaining": next_state.remaining})
        state = next_state
    return moves


def run_samples(output):
    output = Path(output)
    first, second, third, fourth = [load_sample(n) for n in range(1, 5)]
    g1 = Grid.from_rows(first["grid"])
    g2 = Grid.from_rows(second["grid"], marker_costs=second["marker_costs"])
    r1 = astar(g1, tuple(first["start"]), tuple(first["goal"]), record_trace=True)
    r2 = astar(g2, tuple(second["start"]), tuple(second["goal"]), record_trace=True)
    r3 = solve_tree(tree_from_dict(third), record_trace=True)
    assert r1.cost == first["expected_steps"] and len(r1.path) - 1 == 7
    assert r2.cost == second["expected_cost"]
    assert r3.value == third["expected_value"]
    game, matrix, pair_results = make_treasure_game(fourth)
    games, exact_roots, lines = [], {}, {}
    for depth in (2, 4):
        expected = fourth["expected"][str(depth)]
        for algorithm in ("minimax", "alpha_beta"):
            result = solve_treasure(game, depth, algorithm=algorithm, record_trace=True)
            assert result.value == expected["best_value"]
            assert result.best_target == expected["best_target"]
            games.append({"depth": depth, "algorithm": algorithm, **asdict(result)})
            write_json(output / "traces" / f"problem4_h{depth}_{algorithm}.json", result.trace)
        exact_roots[str(depth)] = root_actions(game, depth)
        lines[str(depth)] = optimal_line(game, depth)
    for n, result in ((1, r1), (2, r2), (3, r3)):
        write_json(output / "traces" / f"problem{n}.json", result.trace)
    pairs = [{"source": r["source"], "target": r["target"],
              "cost": r["result"].cost, **asdict(r["result"].stats)} for r in pair_results]
    write_csv(output / "distance_matrix.csv", [dict(zip(["source", *fourth["names"]], [name, *row]))
                                               for name, row in zip(fourth["names"], matrix)])
    write_csv(output / "distance_searches.csv", pairs)
    result = {
        "answers": {"problem1_path": r1.path, "problem1_steps": r1.cost,
                    "problem2_cost": r2.cost, "problem3_value": r3.value,
                    "problem4": [{"depth": r["depth"], "algorithm": r["algorithm"],
                                  "best_target": r["best_target"], "best_value": r["value"]} for r in games]},
        "problem1": {"path": r1.path, "stats": asdict(r1.stats), "accumulation": path_accumulation(g1, r1.path)},
        "problem2": {"path": r2.path, "stats": asdict(r2.stats), "accumulation": path_accumulation(g2, r2.path)},
        "problem3": {"value": r3.value, "stats": asdict(r3.stats)},
        "problem4": {"names": fourth["names"], "distance_matrix": matrix,
                     "exact_root_action_values": exact_roots, "optimal_lines": lines,
                     "searches": [{k: v for k, v in r.items() if k != "trace"} for r in games]},
        "all_passed": True,
    }
    write_json(output / "samples.json", result)
    return result


def describe_result(result):
    if hasattr(result, "cost"):
        return {"cost": result.cost, "path_steps": len(result.path) - 1 if result.path else None}
    return {"value": result.value, "best_target": result.best_target}


class Benchmarks:
    def __init__(self, repeats, warmups, seed):
        self.repeats, self.warmups = repeats, warmups
        self.rng = random.Random(seed)
        self.raw, self.summary = [], []

    def group(self, configurations):
        """配置为(meta, callable)；各轮随机交错，缓存由 solve 内部每次新建。"""
        for _ in range(self.warmups):
            for _, function in configurations:
                function()
        collected = [[] for _ in configurations]
        representatives = {}
        for trial in range(self.repeats):
            indices = list(range(len(configurations)))
            self.rng.shuffle(indices)
            for index in indices:
                meta, function = configurations[index]
                started = perf_counter_ns()
                result = function()
                wall_ns = perf_counter_ns() - started
                stats = asdict(result.stats)
                stats["algorithm_elapsed_ns"] = stats.pop("elapsed_ns")
                row = {**meta, "trial": trial, **describe_result(result), **stats, "wall_ns": wall_ns}
                self.raw.append(row)
                collected[index].append(row)
                representatives[index] = result
        for index, rows in enumerate(collected):
            meta, _ = configurations[index]
            times = [row["wall_ns"] / 1000 for row in rows]
            quartiles = statistics.quantiles(times, n=4, method="inclusive")
            result = representatives[index]
            counters = {k: v for k, v in asdict(result.stats).items() if not k.endswith("_ns")}
            for row in rows:
                if any(row[k] != v for k, v in counters.items()) or any(row[k] != v for k, v in describe_result(result).items()):
                    raise AssertionError(f"相同配置的结果或计数不确定: {meta}")
            self.summary.append({**meta, **describe_result(result), **counters,
                                 "repeats": len(rows), "median_us": statistics.median(times),
                                 "p25_us": quartiles[0], "p75_us": quartiles[2], "min_us": min(times),
                                 "algorithm_median_us": statistics.median(row["algorithm_elapsed_ns"] / 1000 for row in rows)})


def benchmark_samples(bench):
    for problem in (1, 2):
        sample = load_sample(problem)
        grid = Grid.from_rows(sample["grid"], marker_costs=sample.get("marker_costs"))
        start, goal = tuple(sample["start"]), tuple(sample["goal"])
        configs = []
        modes = [("zero", "fifo"), ("manhattan", "fifo"), ("goal_aware", "fifo"), ("goal_aware", "larger_g")]
        for heuristic, tie in modes:
            meta = {"suite": "required", "case": f"problem{problem}", "variant": f"{heuristic}/{tie}",
                    "algorithm": "astar", "heuristic": heuristic, "tie_break": tie,
                    "heuristic_admissible": not (problem == 2 and heuristic == "manhattan")}
            configs.append((meta, lambda g=grid, s=start, t=goal, h=heuristic, tb=tie: astar(g, s, t, h, tb)))
        bench.group(configs)
    tree = tree_from_dict(load_sample(3))
    bench.group([({"suite": "required", "case": "problem3", "variant": f"{algorithm}/{order}",
                  "algorithm": algorithm, "order": order, "cache": False},
                 lambda a=algorithm, o=order: solve_tree(tree, a, o))
                for algorithm in ("minimax", "alpha_beta") for order in ("natural", "reverse")])
    game, _, _ = make_treasure_game(load_sample(4))
    for depth in (1, 2, 3, 4):
        configs = []
        for algorithm in ("minimax", "alpha_beta"):
            for order in ("natural", "reverse", "greedy"):
                for cache in (False, True):
                    meta = {"suite": "required" if depth in (2, 4) else "horizon_extension", "case": "problem4",
                            "variant": f"{algorithm}/{order}/cache={cache}", "algorithm": algorithm,
                            "depth": depth, "order": order, "cache": cache}
                    configs.append((meta, lambda a=algorithm, o=order, c=cache, d=depth:
                                    solve_treasure(game, d, a, o, c)))
        bench.group(configs)


def connected_cells(grid, start=(0, 0)):
    found, queue = {start}, deque([start])
    while queue:
        for neighbor in grid.neighbors(queue.popleft()):
            if neighbor not in found:
                found.add(neighbor)
                queue.append(neighbor)
    return sorted(found)


def generated_grid(kind, seed, size=25):
    rng = random.Random(seed)
    rows = [[1 for _ in range(size)] for _ in range(size)]
    if kind in ("weighted", "zero_cost"):
        for r in range(size):
            for c in range(size):
                choices = (0, 0, 1, 2, 5, 9) if kind == "zero_cost" else (1, 1, 2, 3, 6, 9)
                rows[r][c] = None if rng.random() < 0.20 else rng.choice(choices)
        # 固定连通外边界；内部不保证全连通，查询仅在A所在分量中采样。
        for i in range(size):
            rows[0][i] = rows[-1][i] = rows[i][0] = rows[i][-1] = 1
    elif kind == "walls":
        for c in range(3, size - 1, 4):
            gap = size - 2 if (c // 4) % 2 == 0 else 1
            for r in range(size):
                if r != gap:
                    rows[r][c] = None
    return Grid(tuple(tuple(row) for row in rows))


def benchmark_landmarks(bench, output, seed):
    """12个固定查询共享一次索引；同时给出预处理+在线查询总成本。"""
    cases = [("open", seed), ("walls", seed)]
    cases += [("weighted", seed + offset) for offset in range(3)]
    cases += [("zero_cost", seed + offset) for offset in range(2)]
    inputs, preprocessing, amortized = [], [], []
    for kind, case_seed in cases:
        case = f"{kind}_{case_seed}"
        print(f"  寻路拓展：{case}", flush=True)
        grid = generated_grid(kind, case_seed)
        cells = connected_cells(grid)
        rng = random.Random(case_seed + 31)
        queries = [((0, 0), (grid.rows - 1, grid.cols - 1))]
        queries += [tuple(rng.sample(cells, 2)) for _ in range(11)]
        corners = [(0, 0), (0, grid.cols - 1), (grid.rows - 1, 0), (grid.rows - 1, grid.cols - 1)]
        landmarks = list(dict.fromkeys(min(cells, key=lambda p: abs(p[0] - r) + abs(p[1] - c)) for r, c in corners))
        for _ in range(bench.warmups):
            LandmarkIndex.build(grid, landmarks)
        preparation_times = []
        for trial in range(bench.repeats):
            started = perf_counter_ns()
            index = LandmarkIndex.build(grid, landmarks)
            wall_ns = perf_counter_ns() - started
            preparation_times.append(wall_ns)
            preprocessing.append({"case": case, "trial": trial, "wall_ns": wall_ns,
                                  "algorithm_elapsed_ns": index.preprocessing_ns,
                                  "expanded": index.preprocessing_expanded,
                                  "table_entries": index.table_entries, "landmarks": len(landmarks)})
        inputs.append({"case": case, "kind": kind, "seed": case_seed, "grid": grid.costs,
                       "queries": queries, "landmarks": landmarks})
        for query_id, (start, goal) in enumerate(queries):
            optimum = astar(grid, start, goal, "zero").cost
            configs = []
            for variant, heuristic, tie in [("dijkstra", "zero", "fifo"),
                                             ("goal_aware_fifo", "goal_aware", "fifo"),
                                             ("goal_aware_larger_g", "goal_aware", "larger_g"),
                                             ("alt_larger_g", "alt", "larger_g")]:
                def search(h=heuristic, tb=tie, s=start, t=goal, optimum=optimum):
                    # ALT的目标工厂也纳入外层wall_ns，避免隐藏每目标的O(V)准备。
                    actual_h = index.heuristic(t) if h == "alt" else h
                    result = astar(grid, s, t, actual_h, tb)
                    if result.cost != optimum:
                        raise AssertionError(f"拓展配置最优性不一致: {case}, {s}, {t}, {h}")
                    return result
                configs.append(({"suite": "path_extension", "case": case, "kind": kind,
                                 "seed": case_seed, "query": query_id, "variant": variant,
                                 "algorithm": "astar", "heuristic": heuristic, "tie_break": tie}, search))
            bench.group(configs)
        # 同一trial的12次在线时间相加，再加入该trial的一次索引预处理。
        for variant in ("dijkstra", "goal_aware_fifo", "goal_aware_larger_g", "alt_larger_g"):
            per_trial = []
            for trial in range(bench.repeats):
                online = sum(row["wall_ns"] for row in bench.raw if row["suite"] == "path_extension"
                             and row["case"] == case and row["variant"] == variant and row["trial"] == trial)
                prep = preparation_times[trial] if variant == "alt_larger_g" else 0
                per_trial.append({"case": case, "kind": kind, "variant": variant, "trial": trial,
                                  "query_count": len(queries), "online_ns": online,
                                  "preprocessing_ns": prep, "total_ns": online + prep})
            amortized.extend(per_trial)
    write_json(output / "generated" / "path_inputs.json", inputs)
    write_csv(output / "landmark_preprocessing.csv", preprocessing)
    write_csv(output / "landmark_amortization.csv", amortized)
    return inputs, amortized


def benchmark_game_scaling(bench, output, seed):
    inputs = []
    for count, offset in ((4, 0), (6, 0), (8, 0), (8, 1), (8, 2)):
        case_seed = seed + offset
        case = f"treasures_{count}_seed_{case_seed}"
        print(f"  博弈拓展：{case}", flush=True)
        grid = generated_grid("weighted", case_seed, size=11)
        rng = random.Random(case_seed + count)
        names = ["A", "B", *[f"T{i + 1}" for i in range(count)]]
        positions = dict(zip(names, rng.sample(connected_cells(grid), len(names))))
        # 与原题一样，所有特殊位置进入代价为1，其余地形保留。
        rows = [list(row) for row in grid.costs]
        for r, c in positions.values():
            rows[r][c] = 1
        sample = {"grid": rows, "names": names, "positions": positions,
                  "values": {name: rng.randint(4, 20) for name in names[2:]}}
        game, matrix, _ = make_treasure_game(sample)
        expected = solve_treasure(game, count, algorithm="minimax")
        # 每个根动作独立全窗口搜索，校验并列首步而非强制同一名称。
        root_values = root_actions(game, count)
        inputs.append({"case": case, "seed": case_seed, **sample, "distance_matrix": matrix,
                       "expected_value": expected.value, "exact_root_action_values": root_values})
        configurations = []
        for algorithm, order, cache in [("minimax", "natural", False), ("minimax", "natural", True),
                                         ("alpha_beta", "natural", False), ("alpha_beta", "reverse", False),
                                         ("alpha_beta", "greedy", False), ("alpha_beta", "greedy", True)]:
            def search(a=algorithm, o=order, c=cache):
                result = solve_treasure(game, count, a, o, c)
                if result.value != expected.value or root_values[result.best_target] != expected.value:
                    raise AssertionError(f"博弈拓展结果或首步错误: {case}, {a}, {o}, {c}")
                return result
            configurations.append(({"suite": "game_extension", "case": case, "seed": case_seed,
                                    "treasures": count, "depth": count, "variant": f"{algorithm}/{order}/cache={cache}",
                                    "algorithm": algorithm, "order": order, "cache": cache}, search))
        bench.group(configurations)
    write_json(output / "generated" / "game_inputs.json", inputs)
    return inputs


def benchmark_reuse_batches(output, repeats, warmups, seed):
    """连续实际批次：每批ALT新建一次索引，展示查询量增加时的摊销拐点。"""
    records, inputs = [], []
    rng_order = random.Random(seed + 123)
    for kind in ("walls", "weighted", "zero_cost"):
        print(f"  连续查询摊销：{kind}，Q=1/12/60/120", flush=True)
        grid = generated_grid(kind, seed)
        cells = connected_cells(grid)
        rng = random.Random(seed + 571)
        # 这组全部随机查询，不特殊选择地标作为终点。
        queries = [tuple(rng.sample(cells, 2)) for _ in range(120)]
        corners = [(0, 0), (0, grid.cols - 1), (grid.rows - 1, 0), (grid.rows - 1, grid.cols - 1)]
        landmarks = list(dict.fromkeys(min(cells, key=lambda p: abs(p[0] - r) + abs(p[1] - c)) for r, c in corners))
        expected = [astar(grid, start, goal, "zero").cost for start, goal in queries]
        inputs.append({"case": f"{kind}_{seed}", "grid": grid.costs,
                       "queries": queries, "landmarks": landmarks, "expected_costs": expected})
        for query_count in (1, 12, 60, 120):
            def run(variant):
                started = perf_counter_ns()
                index = LandmarkIndex.build(grid, landmarks) if variant == "alt_larger_g" else None
                prepared = perf_counter_ns()
                expanded, costs = 0, []
                for start, goal in queries[:query_count]:
                    h = index.heuristic(goal) if index else "goal_aware"
                    result = astar(grid, start, goal, h, "larger_g")
                    expanded += result.stats.expanded
                    costs.append(result.cost)
                finished = perf_counter_ns()
                if costs != expected[:query_count]:
                    raise AssertionError("连续查询批次出现次优结果")
                return {"wall_ns": finished - started, "preprocessing_ns": prepared - started if index else 0,
                        "query_phase_ns": finished - prepared, "expanded": expanded,
                        "preprocessing_expanded": index.preprocessing_expanded if index else 0}
            for _ in range(warmups):
                run("goal_aware_larger_g")
                run("alt_larger_g")
            for trial in range(repeats):
                order = ["goal_aware_larger_g", "alt_larger_g"]
                rng_order.shuffle(order)
                for variant in order:
                    records.append({"case": f"{kind}_{seed}", "kind": kind, "query_count": query_count,
                                    "variant": variant, "trial": trial, **run(variant)})
    write_csv(output / "reuse_batches.csv", records)
    write_json(output / "generated" / "reuse_batch_inputs.json", inputs)
    return records


def heuristic_audit(output):
    """直接逐点/逐边检验题二启发式；真距离用独立Bellman-Ford求出。"""
    sample = load_sample(2)
    grid = Grid.from_rows(sample["grid"], marker_costs=sample["marker_costs"])
    goal = tuple(sample["goal"])
    vertices = [(r, c) for r in range(grid.rows) for c in range(grid.cols) if grid.costs[r][c] is not None]
    distances = {node: math.inf for node in vertices}
    distances[goal] = 0
    for _ in range(len(vertices) - 1):
        changed = False
        for node in vertices:
            for nxt in grid.neighbors(node):
                candidate = grid.cost(nxt) + distances[nxt]
                if candidate < distances[node]:
                    distances[node], changed = candidate, True
        if not changed:
            break
    from .pathfinding import goal_aware_heuristic
    heuristics = {"zero": lambda node: 0,
                  "raw_manhattan": lambda node: abs(node[0] - goal[0]) + abs(node[1] - goal[1]),
                  "goal_aware": goal_aware_heuristic(grid, goal)}
    records = []
    for name, h in heuristics.items():
        for node in vertices:
            records.append({"heuristic": name, "row": node[0], "col": node[1], "h": h(node),
                            "true_distance": distances[node], "overestimate": max(0, h(node) - distances[node]),
                            "inconsistent_outgoing_edges": sum(h(node) > grid.cost(nxt) + h(nxt) for nxt in grid.neighbors(node))})
    write_csv(output / "heuristic_audit.csv", records)
    # 这是允许内部零权的扩展反例，与原题输入明确区分。
    counter_rows = [["S", 3, "E"], [1, 0, 1]]
    counter_grid = Grid.from_rows(counter_rows, marker_costs={"E": 0})
    counter_runs = []
    for heuristic in ("zero", "manhattan", "goal_aware"):
        for tie in ("fifo", "larger_g"):
            result = astar(counter_grid, (0, 0), (0, 2), heuristic, tie, record_trace=True)
            expected = 3 if heuristic == "manhattan" else 2
            assert result.cost == expected
            counter_runs.append({"heuristic": heuristic, "tie_break": tie, **asdict(result)})
    write_json(output / "heuristic_counterexample.json", {
        "scope": "拓展：额外允许内部零代价格，不是原题给定地图", "grid": counter_rows,
        "start": [0, 0], "goal": [0, 2], "marker_costs": {"E": 0}, "runs": counter_runs,
    })
    return records


def add_reductions(summary):
    for row in summary:
        if row["algorithm"] not in ("minimax", "alpha_beta"):
            continue
        baseline = next((other for other in summary if other["case"] == row["case"]
                         and other.get("depth") == row.get("depth") and other["algorithm"] == "minimax"
                         and other.get("order") == "natural" and not other.get("cache")), None)
        if baseline:
            row["avoided_visits_vs_minimax"] = baseline["visited"] - row["visited"]
            row["visit_reduction_fraction"] = 1 - row["visited"] / baseline["visited"]


def write_findings(output, samples, summary, amortized, audit, reuse_batches):
    lines = ["# 实验数据摘要（由程序生成）", "", "本文件供后续实验报告使用。耗时为本机重复执行的中位数；完整原始样本保存在 raw_runs.csv。",
             "", "## 文档样例与基础算法", "", "所有给定样例已通过。以下使用默认访问顺序、关闭博弈缓存。", "",
             "| 题目/深度 | 算法 | 结果 | 有效出堆/递归访问 | 展开 | 叶评价 | 剪枝次数 | 直接跳过分支 | 中位耗时 μs |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["suite"] != "required":
            continue
        if row["algorithm"] == "astar" and row["variant"] != "goal_aware/fifo":
            continue
        if row["algorithm"] != "astar" and (row.get("order") != "natural" or row.get("cache")):
            continue
        label = row["case"] + (f" H={row['depth']}" if "depth" in row else "")
        value = row.get("cost", row.get("value"))
        if row.get("best_target"):
            value = f"{row['best_target']} / {value}"
        lines.append(f"| {label} | {row['algorithm']} | {value} | {row.get('popped', row.get('visited'))} | {row['expanded']} | {row.get('leaves', '-')} | {row.get('cutoffs', '-')} | {row.get('pruned_branches', '-')} | {row['median_us']:.2f} |")
    lines += ["", "## 题四首步的精确效用", "", "以下每个动作都独立全窗口搜索，不能从被剪枝的内部返回值直接推定。", "",
              "| 深度 | T1 | T2 | T3 | T4 |", "|---|---:|---:|---:|---:|"]
    for depth, values in samples["problem4"]["exact_root_action_values"].items():
        lines.append(f"| {depth} | " + " | ".join(str(values[f"T{i}"]) for i in range(1, 5)) + " |")
    lines += ["", "## 创新与拓展数据", "", "### A* 同分规则与 ALT", "",
              "| 场景 | 策略 | 12次查询展开合计 | 在线中位总耗时 ms | 含预处理中位总耗时 ms |", "|---|---|---:|---:|---:|"]
    for case in dict.fromkeys(row["case"] for row in amortized):
        for variant in ("dijkstra", "goal_aware_fifo", "goal_aware_larger_g", "alt_larger_g"):
            runs = [r for r in amortized if r["case"] == case and r["variant"] == variant]
            expanded = sum(r["expanded"] for r in summary if r["suite"] == "path_extension" and r["case"] == case and r["variant"] == variant)
            lines.append(f"| {case} | {variant} | {expanded} | {statistics.median(r['online_ns'] for r in runs)/1e6:.3f} | {statistics.median(r['total_ns'] for r in runs)/1e6:.3f} |")
    lines += ["", "ALT 同时报告正反向地标距离表的预处理耗时、展开数和表项数。四个地标取接近四角的可达格；",
              "每图第一个查询使用对角角点（终点也是地标），其余11个查询由固定种子采样，避免只展示有利特例。",
              "同一trial内的12个在线查询耗时相加，另加一次预处理，属于组成项成本核算，并非连续批次墙钟测量。",
              "", "### 连续实际查询批次（每批重新预处理）", "", "此组的120个查询全部随机采样，取相同列表的前Q个，ALT每批新建一次索引。总耗时直接测量，含预处理。", "",
              "| 场景 | Q | 安全曼哈顿+larger-g 总耗时 ms | ALT+larger-g 总耗时 ms |", "|---|---:|---:|---:|"]
    for case in dict.fromkeys(row["case"] for row in reuse_batches):
        for query_count in (1, 12, 60, 120):
            medians = [statistics.median(r["wall_ns"] for r in reuse_batches
                                        if r["case"] == case and r["query_count"] == query_count and r["variant"] == variant)/1e6
                       for variant in ("goal_aware_larger_g", "alt_larger_g")]
            lines.append(f"| {case} | {query_count} | {medians[0]:.3f} | {medians[1]:.3f} |")
    lines += ["", "### 宝藏数量扩展与缓存", "", "| 场景 | 策略 | 访问 | 剪枝 | 跳过直接分支 | 缓存命中 | 访问减少 | 中位耗时 ms |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["suite"] == "game_extension":
            lines.append(f"| {row['case']} | {row['variant']} | {row['visited']} | {row['cutoffs']} | {row['pruned_branches']} | {row['cache_hits']} | {row['visit_reduction_fraction']:.1%} | {row['median_us']/1000:.3f} |")
    violations = sum(row["overestimate"] > 0 for row in audit if row["heuristic"] == "raw_manhattan")
    lines += ["", "## 解读边界与报告写作提醒", "",
              f"- 题二原始曼哈顿在 {violations} 个可达格高估，因此不可采纳。但仅E为0、其余格权重≥1的特殊模型仍可证明返回最优，不能据此把违反充分条件等同于必然失败；详见 docs/ALGORITHMS.md。逐点审计见 heuristic_audit.csv。",
              "- heuristic_counterexample.json 给出允许内部零权的扩展反例：raw Manhattan得到3，安全下界和Dijkstra得到2；这是额外场景，不是改写给定题目。",
              "- 优先比较确定性的节点计数。微秒级耗时受操作系统调度、Python解释器和缓存影响；保留IQR，不用单次运行定胜负。",
              "- 外层wall耗时包括入口校验；题三_validate_tree会访问全树检查输入，虽然不计入搜索visited，仍会稀释剪枝带来的端到端加速。algorithm_elapsed_ns仅计博弈递归，可分开分析。",
              "- larger-g仅在f相同时改变顺序，保持最优性；并非任何地图都节省节点。ALT在多查询时可能摊销预处理，须同时检查总成本。",
              "- 题四只有4个宝藏时状态树小，缓存可能无命中或净变慢。6/8宝藏场景用于检验扩展性，不能把最有利种子代替整体表现。",
              "- greedy仅按当前玩家即时收益排序，不改变极大极小目标，也不预先偷看子树的真实值。",
              "- pruned_branches仅是直接未访问兄弟数；visited相对Minimax的减少包含整棵子树效应，启用缓存时还包含缓存节省。",
              "- 拓展是经典方法的正确实现、组合与对照，不声称提出新算法。创新评分需结合理论、证据与适用条件。", ""]
    (output / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiments(output, repeats=15, warmups=3, seed=20261023, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print("[1/5] 核验四道课例与逐步轨迹", flush=True)
    samples = run_samples(output)
    audit = heuristic_audit(output)
    bench = Benchmarks(repeats, warmups, seed)
    print("[2/5] 原题启发式、访问顺序、搜索深度与缓存对照", flush=True)
    benchmark_samples(bench)
    print("[3/5] 多地图、多查询、地标预处理及摊销实验", flush=True)
    _, amortized = benchmark_landmarks(bench, output, seed)
    reuse_batches = benchmark_reuse_batches(output, repeats, warmups, seed)
    print("[4/5] 4/6/8宝藏规模、排序和缓存消融实验", flush=True)
    benchmark_game_scaling(bench, output, seed)
    add_reductions(bench.summary)
    write_csv(output / "raw_runs.csv", bench.raw)
    write_csv(output / "summary.csv", bench.summary)
    write_json(output / "summary.json", bench.summary)
    write_findings(output, samples, bench.summary, amortized, audit, reuse_batches)
    files = sorted([*ROOT.glob("lab2/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("data/*.json")])
    manifest = {
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "timezone": "Asia/Shanghai",
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "seed": seed, "repeats": repeats, "warmups": warmups,
        "clock": vars(get_clock_info("perf_counter")), "timing_unit": "nanoseconds",
        "raw_runs": len(bench.raw), "configurations": len(bench.summary),
        "reuse_batch_runs": len(reuse_batches),
        "trace_during_benchmark": False, "fresh_game_cache_each_solve": True,
        "file_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
        "requirement_sha256": sha256((ROOT / "实验二要求_new.pdf").read_bytes()).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    print("[5/5] 输出数据摘要与图表", flush=True)
    if plots:
        from .plots import make_plots
        make_plots(output)
    print(f"完成：{len(bench.summary)}种配置，{len(bench.raw)}次计时运行。结果目录：{output.resolve()}", flush=True)
