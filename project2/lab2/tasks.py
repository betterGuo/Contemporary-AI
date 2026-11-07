"""四道题的薄接口：返回文档要求的结果，仪表数据由底层 API 提供。"""

from __future__ import annotations

import json
from pathlib import Path

from .pathfinding import Grid, astar
from .games import TreasureGame, solve_tree, solve_treasure, tree_from_dict

ROOT = Path(__file__).resolve().parents[1]


def load_sample(problem: int) -> dict:
    return json.loads((ROOT / "data" / f"problem{problem}.json").read_text(encoding="utf-8"))


def solve_maze(start, goal, grid):
    """题一：包含起终点的坐标序列，无解返回 []。"""
    return astar(Grid.from_rows(grid), tuple(start), tuple(goal)).path


def solve_weighted(start, goal, grid):
    """题二：进入 E 为 0，起点不收费，无解返回 -1。"""
    result = astar(Grid.from_rows(grid, marker_costs={"E": 0}), tuple(start), tuple(goal))
    return result.cost if result.cost is not None else -1


def solve_game_tree(tree_structure):
    """题三：从左到右执行 Alpha-beta，返回根 MAX 的得分。"""
    return solve_tree(tree_from_dict(tree_structure)).value


def build_distance_matrix(sample):
    """题四：逐个有序点对执行 A*，不能假设一般进入格代价图是对称图。"""
    grid = Grid.from_rows(sample["grid"])
    names, positions = sample["names"], sample["positions"]
    matrix, records = [], []
    for source in names:
        row = []
        for target in names:
            result = astar(grid, tuple(positions[source]), tuple(positions[target]))
            row.append(result.cost)
            records.append({"source": source, "target": target, "result": result})
        matrix.append(row)
    return matrix, records


def make_treasure_game(sample):
    matrix, records = build_distance_matrix(sample)
    return TreasureGame(matrix, sample["names"], sample["values"]), matrix, records


def solve_treasures(sample, depth=4, algorithm="alpha_beta"):
    game, matrix, _ = make_treasure_game(sample)
    result = solve_treasure(game, depth, algorithm=algorithm)
    return {"distance_matrix": matrix, "best_target": result.best_target, "best_value": result.value}
