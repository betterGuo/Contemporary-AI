"""实验二的通用加权网格、A* 与有向 ALT 地标启发函数。

移动代价采用“进入目标格的权重”，起点不收费。即使几何邻接是双向的，
两个方向的边权也可能不同，因此 ALT 必须分别保存正向、反向最短距离。
仅使用 Python 标准库；邻居顺序固定为上、右、下、左，便于复现实验。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from numbers import Real
import re
from time import perf_counter_ns
from types import MappingProxyType


Coord = tuple[int, int]
Heuristic = Callable[[Coord], int | float]
_MARKER = re.compile(r"(?:\.|S|E|A|B|T\d+)")
_DIGITS = re.compile(r"[0-9]+")
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _nonnegative_integer(value: object, label: str) -> int:
    """拒绝 bool、浮点数和负权重，避免 Dijkstra/A* 的前提被悄悄破坏。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} 必须是非负整数，实际为 {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Grid:
    """只读矩形网格；None 是障碍，非负整数是进入该格的代价。"""

    costs: tuple[tuple[int | None, ...], ...]

    def __post_init__(self) -> None:
        try:
            normalized = tuple(tuple(row) for row in self.costs)
        except TypeError as exc:
            raise ValueError("网格必须是非空的二维矩形序列") from exc
        if not normalized or not normalized[0]:
            raise ValueError("网格不能为空")
        width = len(normalized[0])
        for row in normalized:
            if len(row) != width:
                raise ValueError("网格的每一行必须等长")
            for value in row:
                if value is not None:
                    _nonnegative_integer(value, "网格权重")
        object.__setattr__(self, "costs", normalized)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Sequence[object] | str],
        marker_costs: Mapping[str, int] | None = None,
    ) -> Grid:
        """解析实验输入，保留数字格的原权重。

        ``X`` / ``None`` 是障碍；``. S E A B T1 T2 ...`` 默认权重 1。
        题二应显式传入 ``marker_costs={"E": 0}``，题四则保留默认值。
        每行可为 token 序列、空格分隔字符串，或仅含单字符格的字符串。
        多位数字和 ``T1`` 等多字符标记必须采用 token 或空格分隔形式。
        """
        overrides = dict(marker_costs or {})
        for marker, value in overrides.items():
            if not isinstance(marker, str) or not _MARKER.fullmatch(marker):
                raise ValueError(f"未知的权重覆盖标记: {marker!r}")
            _nonnegative_integer(value, f"标记 {marker} 的权重")
        if isinstance(rows, (str, bytes)):
            raise ValueError("请传入行序列；多行文本可先调用 splitlines()")

        parsed: list[tuple[int | None, ...]] = []
        try:
            row_iterator = iter(rows)
        except TypeError as exc:
            raise ValueError("网格必须是行序列") from exc
        for row in row_iterator:
            if isinstance(row, str):
                stripped = row.strip()
                cells = stripped.split() if any(c.isspace() for c in stripped) else list(stripped)
            else:
                try:
                    cells = list(row)
                except TypeError as exc:
                    raise ValueError("网格的每一行必须是 token 序列或字符串") from exc
            parsed_row: list[int | None] = []
            for cell in cells:
                if cell is None or (isinstance(cell, str) and cell == "X"):
                    parsed_row.append(None)
                elif isinstance(cell, str):
                    if _DIGITS.fullmatch(cell):
                        parsed_row.append(int(cell))
                    elif _MARKER.fullmatch(cell):
                        parsed_row.append(overrides.get(cell, 1))
                    else:
                        raise ValueError(f"未知的网格标记: {cell!r}")
                else:
                    parsed_row.append(_nonnegative_integer(cell, "网格权重"))
            parsed.append(tuple(parsed_row))
        return cls(tuple(parsed))

    @property
    def rows(self) -> int:
        return len(self.costs)

    @property
    def cols(self) -> int:
        return len(self.costs[0])

    def _check_coord(self, pos: Coord) -> None:
        if (
            not isinstance(pos, tuple)
            or len(pos) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in pos)
            or not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols)
        ):
            raise ValueError(f"坐标必须是网格范围内的 (行, 列) 整数元组: {pos!r}")

    def cost(self, pos: Coord) -> int:
        """返回进入格子的代价；越界坐标或障碍格属于无效输入。"""
        self._check_coord(pos)
        value = self.costs[pos[0]][pos[1]]
        if value is None:
            raise ValueError(f"坐标 {pos} 是障碍格")
        return value

    def neighbors(self, pos: Coord) -> Iterable[Coord]:
        """按上、右、下、左枚举合法的四连通邻居。"""
        self.cost(pos)
        row, col = pos
        for dr, dc in _DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.costs[nr][nc] is not None:
                yield nr, nc


def goal_aware_heuristic(grid: Grid, goal: Coord) -> Heuristic:
    """构建可采纳且一致的、适用于零权终点的曼哈顿下界。

    对非终点 n，令 M 为曼哈顿距离，m 为所有非终点可走格子的最小权重，
    则 h(n) = (M - 1) * m + cost(goal)，h(goal) = 0。
    任意到达目标的路径至少有 M 条边，其中最后一条必收取 cost(goal)。
    其余进入非终点的边均至少收取 m；因此即使目标权重为 0，也不会高估。
    """
    goal_cost = grid.cost(goal)
    minimum = min(
        (
            value
            for row, values in enumerate(grid.costs)
            for col, value in enumerate(values)
            if value is not None and (row, col) != goal
        ),
        default=0,
    )

    def estimate(pos: Coord) -> int:
        if pos == goal:
            return 0
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        return (distance - 1) * minimum + goal_cost

    return estimate


@dataclass(slots=True)
class SearchStats:
    """A* 计数口径；所有计数均属于本次调用。

    popped: 有效出堆次数，包含终点；expanded: 实际枚举邻居的次数，不含终点。
    generated: 枚举合法邻居的总次数（含重复）；discovered: 曾被发现的不同格数。
    reopened: 已关闭格因更小 g 被重新打开的次数；stale_pops: 丢弃过期堆项数。
    max_frontier: 实际堆项数峰值，包含尚未被弹出的过期项。
    elapsed_ns: 包括验证、启发准备、搜索和路径还原的总耗时；
    preparation_ns 是其中验证与启发准备的子集，不能再次加到 elapsed_ns。
    """

    popped: int = 0
    expanded: int = 0
    generated: int = 0
    discovered: int = 0
    reopened: int = 0
    stale_pops: int = 0
    max_frontier: int = 0
    elapsed_ns: int = 0
    preparation_ns: int = 0


@dataclass(slots=True)
class PathResult:
    """无路时 path=[]、cost=None；有效路径包含起点与终点。"""

    path: list[Coord]
    cost: int | None
    stats: SearchStats
    trace: list[dict[str, object]] = field(default_factory=list)


def astar(
    grid: Grid,
    start: Coord,
    goal: Coord,
    heuristic: str | Heuristic = "goal_aware",
    tie_break: str = "fifo",
    record_trace: bool = False,
) -> PathResult:
    """A* 最短路径搜索，支持延迟删除、重新打开和可复现的平局规则。

    ``zero`` 为 Dijkstra 基线；``goal_aware`` 是安全默认值。
    ``manhattan`` 是未经权重修正的原始距离，只用于不安全启发函数对照实验，
    含零代价格时可能高估，从而不保证最优。
    自定义函数须返回有限非负数，且 h(goal)=0；其可采纳性由调用者保证。
    对可采纳但不一致的函数，通过 reopening 保持最优性。
    ``larger_g`` 在 f 相同时优先更大的 g，其后仍按入堆先后排序。
    trace 记录每次有效 pop 的 node/g/h/f 与是否真正 expanded。
    """
    started = perf_counter_ns()
    grid.cost(start)
    grid.cost(goal)
    if tie_break not in {"fifo", "larger_g"}:
        raise ValueError(f"未知的平局处理策略: {tie_break!r}")
    if callable(heuristic):
        estimate = heuristic
    elif heuristic == "zero":
        estimate = lambda pos: 0
    elif heuristic == "manhattan":
        estimate = lambda pos: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    elif heuristic == "goal_aware":
        estimate = goal_aware_heuristic(grid, goal)
    else:
        raise ValueError(f"未知的启发函数: {heuristic!r}")

    heuristic_cache: dict[Coord, int | float] = {}

    def h(pos: Coord) -> int | float:
        if pos not in heuristic_cache:
            value = estimate(pos)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or value < 0
                or (not isinstance(value, int) and not isfinite(value))
            ):
                raise ValueError(f"启发函数必须返回有限非负数，h({pos})={value!r}")
            if pos == goal and value != 0:
                raise ValueError("启发函数必须满足 h(goal)=0")
            heuristic_cache[pos] = value
        return heuristic_cache[pos]

    # 在搜索前检查目标值，避免未遇到目标时隐藏不合约的自定义函数。
    h(goal)
    start_h = h(start)
    stats = SearchStats(discovered=1, max_frontier=1)
    stats.preparation_ns = perf_counter_ns() - started
    trace: list[dict[str, object]] = []
    serial = count()
    frontier: list[tuple[int | float, int, int, int, Coord]] = [
        (start_h, 0, next(serial), 0, start)
    ]
    best_g: dict[Coord, int] = {start: 0}
    parent: dict[Coord, Coord] = {}
    closed: set[Coord] = set()

    while frontier:
        f_score, _, _, current_g, current = heappop(frontier)
        if current_g != best_g[current]:
            stats.stale_pops += 1
            continue
        stats.popped += 1
        is_goal = current == goal
        if record_trace:
            trace.append(
                {"event": "pop", "node": current, "g": current_g, "h": h(current),
                 "f": f_score, "expanded": not is_goal}
            )
        if is_goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            stats.elapsed_ns = perf_counter_ns() - started
            return PathResult(path, current_g, stats, trace)

        closed.add(current)
        stats.expanded += 1
        for neighbor in grid.neighbors(current):
            stats.generated += 1
            candidate_g = current_g + grid.cost(neighbor)
            previous_g = best_g.get(neighbor)
            # 严格改善使零代价环不会造成无穷重入，也防止父指针出现环。
            if previous_g is not None and candidate_g >= previous_g:
                continue
            if previous_g is None:
                stats.discovered += 1
            if neighbor in closed:
                closed.remove(neighbor)
                stats.reopened += 1
            best_g[neighbor] = candidate_g
            parent[neighbor] = current
            secondary = -candidate_g if tie_break == "larger_g" else 0
            heappush(
                frontier,
                (candidate_g + h(neighbor), secondary, next(serial), candidate_g, neighbor),
            )
            stats.max_frontier = max(stats.max_frontier, len(frontier))

    stats.elapsed_ns = perf_counter_ns() - started
    return PathResult([], None, stats, trace)


def _dijkstra_all(grid: Grid, start: Coord, reverse: bool) -> tuple[dict[Coord, int], int]:
    """计算一对多距离；反图边 v→u 的代价必须为原边 u→v 的 cost(v)。"""
    distance: dict[Coord, int] = {start: 0}
    heap: list[tuple[int, Coord]] = [(0, start)]
    expanded = 0
    while heap:
        current_distance, current = heappop(heap)
        if current_distance != distance[current]:
            continue
        expanded += 1
        for neighbor in grid.neighbors(current):
            edge_cost = grid.cost(current) if reverse else grid.cost(neighbor)
            candidate = current_distance + edge_cost
            previous = distance.get(neighbor)
            if previous is None or candidate < previous:
                distance[neighbor] = candidate
                heappush(heap, (candidate, neighbor))
    return distance, expanded


@dataclass(frozen=True, slots=True)
class LandmarkIndex:
    """有向 ALT 预处理结果，可复用于同一网格的多次终点查询。

    forward[i][v] = d(L_i, v)，reverse[i][v] = d(v, L_i)。
    preprocessing_expanded 为所有正反向 Dijkstra 的有效展开总次数。
    table_entries 是保存的距离标量数量，不将其误报为内存字节数。
    预处理时间不属于后续单次 astar.stats.elapsed_ns，应在实验中单独列出，
    并以 preprocessing_ns + 查询耗时总和报告一次性总成本与摊销成本。
    """

    grid: Grid
    landmarks: tuple[Coord, ...]
    forward: tuple[Mapping[Coord, int], ...]
    reverse: tuple[Mapping[Coord, int], ...]
    preprocessing_ns: int
    preprocessing_expanded: int
    table_entries: int

    @classmethod
    def build(cls, grid: Grid, landmarks: Iterable[Coord]) -> LandmarkIndex:
        """为给定地标建立双向距离表；重复地标按首次出现顺序去重。"""
        started = perf_counter_ns()
        unique: list[Coord] = []
        seen: set[Coord] = set()
        for landmark in landmarks:
            grid.cost(landmark)
            if landmark not in seen:
                unique.append(landmark)
                seen.add(landmark)
        forward: list[Mapping[Coord, int]] = []
        reverse: list[Mapping[Coord, int]] = []
        expanded = 0
        entries = 0
        for landmark in unique:
            f_dist, f_expanded = _dijkstra_all(grid, landmark, reverse=False)
            r_dist, r_expanded = _dijkstra_all(grid, landmark, reverse=True)
            forward.append(MappingProxyType(f_dist))
            reverse.append(MappingProxyType(r_dist))
            expanded += f_expanded + r_expanded
            entries += len(f_dist) + len(r_dist)
        return cls(
            grid, tuple(unique), tuple(forward), tuple(reverse),
            perf_counter_ns() - started, expanded, entries,
        )

    def heuristic(self, goal: Coord) -> Heuristic:
        """用有向三角不等式和安全网格下界共同构造一致启发函数。

        h(n) = max(0, d(L,g)-d(L,n), d(n,L)-d(g,L), h_goal_aware(n))。
        只有两个距离均可达时才参与差值计算，避免 inf-inf 产生 NaN。
        此网格的边双向存在，因此一个连通分量内不会出现可达性单向变化。
        ``heuristic(goal)`` 本身的构造耗时也应计入外层查询批次总计时。
        """
        base = goal_aware_heuristic(self.grid, goal)
        goal_forward = tuple(dist.get(goal) for dist in self.forward)
        goal_reverse = tuple(dist.get(goal) for dist in self.reverse)

        def estimate(pos: Coord) -> int:
            bound = base(pos)
            for f_dist, r_dist, from_landmark, to_landmark in zip(
                self.forward, self.reverse, goal_forward, goal_reverse,
            ):
                from_node = f_dist.get(pos)
                to_node = r_dist.get(pos)
                if from_landmark is not None and from_node is not None:
                    bound = max(bound, from_landmark - from_node)
                if to_landmark is not None and to_node is not None:
                    bound = max(bound, to_node - to_landmark)
            return bound

        return estimate
