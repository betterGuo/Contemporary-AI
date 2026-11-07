"""实验二第三、四题：Minimax、Alpha-beta 与带界类型的置换表。

统计口径：visited 为递归调用次数（含缓存命中）；expanded 为实际展开的
内部节点；leaves 为实际评价次数；cutoffs 只计确实跳过兄弟分支的截断；
pruned_branches 仅计直接跳过的兄弟分支，不冒充完整子树的节点数。
每次求解使用新缓存，elapsed_ns 只覆盖搜索，计时不包含输入验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from numbers import Integral
from time import perf_counter_ns
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class SearchStats:
    """两种博弈搜索共用的可复现统计口径。"""

    visited: int = 0
    expanded: int = 0
    leaves: int = 0
    cutoffs: int = 0
    pruned_branches: int = 0
    cache_hits: int = 0
    cache_stores: int = 0
    elapsed_ns: int = 0


@dataclass(slots=True)
class GameResult:
    """求解结果；value 是根状态累计效用，best_target 是根的首步。"""

    value: int
    best_target: str | None
    stats: SearchStats
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TreeNode:
    """博弈树节点；叶节点填写 value，内部节点填写 children。"""

    value: int | None = None
    children: list[TreeNode] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []
        else:
            self.children = list(self.children)


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{label} 必须是整数")
    return int(value)


def tree_from_dict(data: Mapping[str, Any]) -> TreeNode:
    """读取文档 JSON，检查声明的 MAX/MIN 层次与根 MAX 的轮换一致。

    支持 ``{'tree_structure': {...}}`` 包装和裸节点。叶节点为
    ``{'value': 3}``；内部节点包含 children，可以声明布尔字段
    is_maximizing_player。JSON 中复制的相同子树分别计数。
    """

    if not isinstance(data, Mapping):
        raise ValueError("博弈树输入必须是字典")
    root_data = data.get("tree_structure", data)
    active: set[int] = set()

    def parse(node: Any, maximizing: bool) -> TreeNode:
        if not isinstance(node, Mapping):
            raise ValueError("每个博弈树节点必须是字典")
        if id(node) in active:
            raise ValueError("博弈树不能包含循环引用")
        active.add(id(node))
        try:
            if "is_maximizing_player" in node:
                declared = node["is_maximizing_player"]
                if not isinstance(declared, bool) or declared != maximizing:
                    raise ValueError("is_maximizing_player 必须从根 MAX 开始逐层轮换")
            children = node.get("children", [])
            if not isinstance(children, (list, tuple)):
                raise ValueError("children 必须是列表")
            if children:
                if node.get("value") is not None:
                    raise ValueError("内部节点不能同时指定叶节点评价值")
                return TreeNode(children=[parse(child, not maximizing) for child in children])
            return TreeNode(value=_integer(node.get("value"), "叶节点 value"))
        finally:
            active.remove(id(node))

    return parse(root_data, True)


def _validate_tree(root: TreeNode) -> None:
    active: set[int] = set()

    def visit(node: TreeNode) -> None:
        if not isinstance(node, TreeNode):
            raise ValueError("children 中的元素必须是 TreeNode")
        if id(node) in active:
            raise ValueError("博弈树不能包含循环引用")
        active.add(id(node))
        try:
            if node.children:
                if node.value is not None:
                    raise ValueError("内部节点不能同时指定 value")
                for child in node.children:
                    visit(child)
            else:
                _integer(node.value, "叶节点 value")
        finally:
            active.remove(id(node))

    visit(root)


def _check_algorithm(algorithm: str) -> None:
    if algorithm not in {"minimax", "alpha_beta"}:
        raise ValueError("algorithm 必须是 'minimax' 或 'alpha_beta'")


def _bound_for_trace(value: int | float) -> int | str:
    """让 trace 可以直接保存成严格 JSON，避免 Infinity 浮点扩展。"""
    if value == inf:
        return "+inf"
    if value == -inf:
        return "-inf"
    return int(value)


def solve_tree(
    root: TreeNode,
    algorithm: str = "alpha_beta",
    order: str = "natural",
    record_trace: bool = False,
) -> GameResult:
    """根为 MAX，按自然/逆序遍历；不使用缓存或偷看子树真值来排序。

    相同 TreeNode 被多个父节点引用时，仍按树的各次访问分别统计。
    第三题没有宝藏名称，因此 best_target 为 None；trace 的 path 使用
    原始 children 的零基索引，可以准确复核左右访问顺序。
    """
    _check_algorithm(algorithm)
    if order not in {"natural", "reverse"}:
        raise ValueError("博弈树 order 必须是 'natural' 或 'reverse'")
    _validate_tree(root)
    stats = SearchStats()
    trace: list[dict[str, Any]] = []
    pruning = algorithm == "alpha_beta"

    def search(node: TreeNode, maximizing: bool, alpha: int | float,
               beta: int | float, path: tuple[int, ...]) -> int:
        stats.visited += 1
        if record_trace:
            trace.append({"event": "visit", "path": list(path),
                          "player": "MAX" if maximizing else "MIN",
                          "alpha": _bound_for_trace(alpha),
                          "beta": _bound_for_trace(beta)})
        if not node.children:
            stats.leaves += 1
            result = int(node.value)  # 输入已经验证。
            if record_trace:
                trace.append({"event": "evaluate", "path": list(path), "value": result})
            return result
        stats.expanded += 1
        indices = list(range(len(node.children)))
        if order == "reverse":
            indices.reverse()
        best: int | float = -inf if maximizing else inf
        for offset, index in enumerate(indices):
            value = search(node.children[index], not maximizing, alpha, beta, path + (index,))
            best = max(best, value) if maximizing else min(best, value)
            if pruning:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                skipped = len(indices) - offset - 1
                if alpha >= beta and skipped:
                    stats.cutoffs += 1
                    stats.pruned_branches += skipped
                    if record_trace:
                        trace.append({"event": "cutoff", "path": list(path),
                                      "alpha": _bound_for_trace(alpha),
                                      "beta": _bound_for_trace(beta),
                                      "skipped_children": indices[offset + 1:]})
                    break
        return int(best)

    started = perf_counter_ns()
    value = search(root, True, -inf, inf, ())
    stats.elapsed_ns = perf_counter_ns() - started
    return GameResult(value, None, stats, trace)


@dataclass(frozen=True, slots=True)
class GameState:
    """宝藏博弈状态；分数为累计分数，remaining 只含未领取宝藏名称。"""

    pos_a: str
    pos_b: str
    remaining: tuple[str, ...]
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining", tuple(self.remaining))
        object.__setattr__(self, "score_a", _integer(self.score_a, "score_a"))
        object.__setattr__(self, "score_b", _integer(self.score_b, "score_b"))
        if self.player not in {"A", "B"}:
            raise ValueError("player 必须是 'A' 或 'B'")

    @property
    def utility(self) -> int:
        """A 视角累计效用 U = ScoreA - ScoreB。"""
        return self.score_a - self.score_b


class TreasureGame:
    """由 A* 距离矩阵驱动的宝藏分配博弈，不依赖寻路实现。

    names 与距离矩阵行列严格对应，必须有 A/B，其他名称都是宝藏；
    values 只含宝藏的整数价值。距离 None 表示不可达。每次只领取所选
    目标，移动途中经过其他宝藏不领取。负即时收益也属于合法动作。
    """

    def __init__(self, distance_matrix: Sequence[Sequence[int | None]],
                 names: Sequence[str], values: Mapping[str, int]) -> None:
        self.names = tuple(names)
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("位置名称必须是非空字符串")
        if len(set(self.names)) != len(self.names):
            raise ValueError("位置名称不能重复")
        if "A" not in self.names or "B" not in self.names:
            raise ValueError("names 必须包含起点 A 和 B")
        self.treasures = tuple(name for name in self.names if name not in {"A", "B"})
        if set(values) != set(self.treasures):
            raise ValueError("values 的键必须恰好是 names 中的全部宝藏")
        self.values = {name: _integer(values[name], f"{name} 的价值") for name in self.treasures}
        self._indices = {name: i for i, name in enumerate(self.names)}
        if len(distance_matrix) != len(self.names):
            raise ValueError("距离矩阵行数必须等于 names 长度")
        rows: list[tuple[int | None, ...]] = []
        for row in distance_matrix:
            if len(row) != len(self.names):
                raise ValueError("距离矩阵必须为正方形")
            checked: list[int | None] = []
            for distance in row:
                if distance is None:
                    checked.append(None)
                else:
                    distance = _integer(distance, "移动距离")
                    if distance < 0:
                        raise ValueError("移动距离必须非负，不可达请使用 None")
                    checked.append(distance)
            rows.append(tuple(checked))
        self.distance_matrix = tuple(rows)

    def initial_state(self) -> GameState:
        """A 先手，双方在各自初始位置，累计分数均为零。"""
        return GameState("A", "B", self.treasures)

    def validate_state(self, state: GameState) -> None:
        """验证外部传入的中间状态。"""
        if not isinstance(state, GameState):
            raise ValueError("state 必须是 GameState")
        if state.pos_a not in self._indices or state.pos_b not in self._indices:
            raise ValueError("玩家位置必须存在于 names 中")
        if len(set(state.remaining)) != len(state.remaining):
            raise ValueError("剩余宝藏不能重复")
        if not set(state.remaining) <= set(self.treasures):
            raise ValueError("remaining 必须只含本游戏中的宝藏")

    def distance(self, start: str, target: str) -> int | None:
        """读取有向移动距离；不假定矩阵对称。"""
        return self.distance_matrix[self._indices[start]][self._indices[target]]

    def action_gain(self, state: GameState, target: str) -> int:
        """当前玩家的即时收益：目标价值减去本次移动代价。"""
        if target not in state.remaining:
            raise ValueError(f"目标 {target!r} 不是尚未领取的宝藏")
        start = state.pos_a if state.player == "A" else state.pos_b
        distance = self.distance(start, target)
        if distance is None:
            raise ValueError(f"无法从 {start!r} 到达 {target!r}")
        return self.values[target] - distance

    def legal_actions(self, state: GameState, order: str = "natural") -> tuple[str, ...]:
        """natural 为输入宝藏顺序；greedy 按当前玩家即时收益降序。

        A/B 都优先访问自己的高即时收益目标，相等时保留输入顺序。
        该排序只使用当前状态数据，不读取任何子树真实 Minimax 值。
        """
        if order not in {"natural", "reverse", "greedy"}:
            raise ValueError("order 必须是 'natural'、'reverse' 或 'greedy'")
        start = state.pos_a if state.player == "A" else state.pos_b
        remaining = set(state.remaining)
        actions = [target for target in self.treasures
                   if target in remaining and self.distance(start, target) is not None]
        if order == "reverse":
            actions.reverse()
        elif order == "greedy":
            actions.sort(key=lambda target: -self.action_gain(state, target))
        return tuple(actions)

    def successor(self, state: GameState, target: str) -> GameState:
        """执行一次取宝行动并换手，位置移动到目标，不自动领取途经宝藏。"""
        gain = self.action_gain(state, target)
        remaining = tuple(name for name in state.remaining if name != target)
        if state.player == "A":
            return GameState(target, state.pos_b, remaining,
                             state.score_a + gain, state.score_b, "B")
        return GameState(state.pos_a, target, remaining,
                         state.score_a, state.score_b + gain, "A")


@dataclass(frozen=True, slots=True)
class _TTEntry:
    value: int
    flag: str
    best_target: str | None


def solve_treasure(
    game: TreasureGame,
    depth: int,
    algorithm: str = "alpha_beta",
    order: str = "natural",
    use_cache: bool = False,
    state: GameState | None = None,
    record_trace: bool = False,
) -> GameResult:
    """以指定行动深度求解宝藏博弈，返回 A 视角累计效用。

    depth 是单个玩家的行动次数。depth=0 或宝藏已取完时评价当前 U。
    扩展约定：当前玩家没有可达宝藏时直接终局，不引入文档未定义的
    跳过回合规则；这不影响全部宝藏可达的题目样例。

    可选创新：每次求解新建置换表。键包含剩余深度、双方位置、剩余
    宝藏和轮到的玩家。缓存值是“未来效用增量”，去掉与后续决策无关的
    已累计 U；沿边递归时同步平移 alpha/beta 窗口，保持数学等价。
    Alpha-beta 记录 EXACT/LOWER/UPPER，缓存剪枝界绝不当成精确值。
    只在严格更优时换首步，避免 fail-soft 的同值界覆盖已验证最佳动作。
    """
    _check_algorithm(algorithm)
    depth = _integer(depth, "搜索深度")
    if depth < 0:
        raise ValueError("搜索深度必须非负")
    if order not in {"natural", "reverse", "greedy"}:
        raise ValueError("order 必须是 'natural'、'reverse' 或 'greedy'")
    current = game.initial_state() if state is None else state
    game.validate_state(current)
    stats = SearchStats()
    trace: list[dict[str, Any]] = []
    table: dict[tuple[Any, ...], _TTEntry] = {}
    pruning = algorithm == "alpha_beta"

    def store(key: tuple[Any, ...], value: int, flag: str, target: str | None) -> None:
        if use_cache:
            table[key] = _TTEntry(value, flag, target)
            stats.cache_stores += 1

    def search(node: GameState, left: int, alpha: int | float,
               beta: int | float, path: tuple[str, ...]) -> tuple[int, str | None]:
        stats.visited += 1
        key = (left, node.pos_a, node.pos_b,
               tuple(name for name in game.treasures if name in node.remaining), node.player)
        original_alpha, original_beta = alpha, beta
        if record_trace:
            trace.append({"event": "visit", "path": list(path), "depth_left": left,
                          "player": node.player, "pos_a": node.pos_a, "pos_b": node.pos_b,
                          "remaining": list(node.remaining), "score_a": node.score_a,
                          "score_b": node.score_b, "utility": node.utility,
                          "alpha_future": _bound_for_trace(alpha),
                          "beta_future": _bound_for_trace(beta)})
        if use_cache and key in table:
            entry = table[key]
            stats.cache_hits += 1
            if record_trace:
                trace.append({"event": "cache_hit", "path": list(path),
                              "flag": entry.flag, "future_value": entry.value})
            if entry.flag == "EXACT":
                return entry.value, entry.best_target
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.value)
            else:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value, entry.best_target

        actions = () if left == 0 or not node.remaining else game.legal_actions(node, order)
        if not actions:
            stats.leaves += 1
            store(key, 0, "EXACT", None)
            if record_trace:
                reason = ("depth_limit" if left == 0 else
                          "all_collected" if not node.remaining else "no_reachable_target")
                trace.append({"event": "evaluate", "path": list(path), "reason": reason,
                              "future_value": 0, "utility": node.utility})
            return 0, None

        stats.expanded += 1
        maximizing = node.player == "A"
        best: int | float = -inf if maximizing else inf
        best_target: str | None = None
        for offset, target in enumerate(actions):
            child = game.successor(node, target)
            signed_gain = child.utility - node.utility
            if pruning:
                child_alpha, child_beta = alpha - signed_gain, beta - signed_gain
            else:
                child_alpha, child_beta = -inf, inf
            future, _ = search(child, left - 1, child_alpha, child_beta, path + (target,))
            value = signed_gain + future
            improves = value > best if maximizing else value < best
            if improves:
                best, best_target = value, target
            if pruning:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                skipped = len(actions) - offset - 1
                if alpha >= beta and skipped:
                    stats.cutoffs += 1
                    stats.pruned_branches += skipped
                    if record_trace:
                        trace.append({"event": "cutoff", "path": list(path),
                                      "alpha_future": _bound_for_trace(alpha),
                                      "beta_future": _bound_for_trace(beta),
                                      "skipped_targets": list(actions[offset + 1:])})
                    break

        value = int(best)
        if not pruning:
            flag = "EXACT"
        elif value <= original_alpha:
            flag = "UPPER"
        elif value >= original_beta:
            flag = "LOWER"
        else:
            flag = "EXACT"
        store(key, value, flag, best_target)
        if record_trace:
            trace.append({"event": "return", "path": list(path), "future_value": value,
                          "utility": node.utility + value, "flag": flag,
                          "best_target": best_target})
        return value, best_target

    started = perf_counter_ns()
    future, best_target = search(current, depth, -inf, inf, ())
    stats.elapsed_ns = perf_counter_ns() - started
    return GameResult(current.utility + future, best_target, stats, trace)
