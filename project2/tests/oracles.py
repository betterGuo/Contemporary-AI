"""Small reference implementations deliberately independent of lab2 algorithms.

The reference shortest-path algorithm is an O(V^2) scan-based Dijkstra,
without the production heap, neighbors(), heuristics, or relaxation helpers.
The game references exhaust every legal continuation without alpha-beta/TT.
"""

from math import inf


MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]

TERRAIN = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]

TREASURE_MAP = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
TREASURE_POSITIONS = [(0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0)]
TREASURE_NAMES = ["A", "B", "T1", "T2", "T3", "T4"]
TREASURE_VALUES = [9, 6, 7, 7]


def parsed_costs(rows, marker_costs=None):
    marker_costs = marker_costs or {}
    result = {}
    for r, row in enumerate(rows):
        for c, token in enumerate(row):
            if token is None or token == "X":
                continue
            if isinstance(token, int):
                result[r, c] = token
            elif isinstance(token, str) and token.isdecimal():
                result[r, c] = int(token)
            else:
                result[r, c] = marker_costs.get(token, 1)
    return result


def reference_distances(rows, start, marker_costs=None):
    costs = parsed_costs(rows, marker_costs)
    distances = {pos: inf for pos in costs}
    distances[start] = 0
    remaining = set(costs)
    while remaining:
        current = min(remaining, key=distances.__getitem__)
        if distances[current] == inf:
            break
        remaining.remove(current)
        r, c = current
        for other in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
            if other in remaining:
                distances[other] = min(distances[other], distances[current] + costs[other])
    return distances


def reference_matrix(rows, positions):
    all_distances = [reference_distances(rows, source) for source in positions]
    return [[distances[target] for target in positions] for distances in all_distances]


def reference_tree(node, maximizing=True):
    if not node.children:
        return node.value
    values = [reference_tree(child, not maximizing) for child in node.children]
    return (max if maximizing else min)(values)


def reference_treasure(matrix, values, depth, pos_a=0, pos_b=1,
                       remaining=None, score_a=0, score_b=0, player="A"):
    """Return exact value and the complete set of optimal matrix target indices."""
    if remaining is None:
        remaining = tuple(range(2, len(matrix)))
    if depth == 0 or not remaining:
        return score_a - score_b, set()
    outcomes = {}
    position = pos_a if player == "A" else pos_b
    for target in remaining:
        distance = matrix[position][target]
        if distance == inf:
            continue
        gain = values[target - 2] - distance
        rest = tuple(t for t in remaining if t != target)
        if player == "A":
            value, _ = reference_treasure(matrix, values, depth - 1, target, pos_b,
                                          rest, score_a + gain, score_b, "B")
        else:
            value, _ = reference_treasure(matrix, values, depth - 1, pos_a, target,
                                          rest, score_a, score_b + gain, "A")
        outcomes[target] = value
    if not outcomes:
        return score_a - score_b, set()
    value = (max if player == "A" else min)(outcomes.values())
    return value, {target for target in outcomes if outcomes[target] == value}
