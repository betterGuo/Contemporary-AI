"""Independent example, randomized, invariant, and boundary tests for A*."""

from math import inf
import random
import unittest

from lab2.pathfinding import Grid, LandmarkIndex, astar, goal_aware_heuristic
from tests.oracles import (
    MAZE, TERRAIN, TREASURE_MAP, TREASURE_POSITIONS,
    parsed_costs, reference_distances, reference_matrix,
)


class PathfindingTests(unittest.TestCase):
    def assert_valid_result(self, rows, start, goal, result, marker_costs=None):
        costs = parsed_costs(rows, marker_costs)
        expected = reference_distances(rows, start, marker_costs)[goal]
        if expected == inf:
            self.assertIsNone(result.cost)
            self.assertEqual(result.path, [])
            return
        self.assertEqual(result.cost, expected)
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], goal)
        self.assertEqual(len(result.path), len(set(result.path)))
        self.assertTrue(all(position in costs for position in result.path))
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                            for a, b in zip(result.path, result.path[1:])))
        self.assertEqual(sum(costs[position] for position in result.path[1:]), expected)
        self.assertGreaterEqual(result.stats.popped, 1)
        self.assertGreaterEqual(result.stats.elapsed_ns, 0)
        self.assertGreaterEqual(result.stats.expanded, 0)
        self.assertGreaterEqual(result.stats.discovered, len(result.path))

    def test_problem1_given_example(self):
        grid = Grid.from_rows(MAZE)
        for heuristic in ("zero", "manhattan", "goal_aware"):
            with self.subTest(heuristic=heuristic):
                result = astar(grid, (0, 0), (3, 4), heuristic=heuristic)
                self.assert_valid_result(MAZE, (0, 0), (3, 4), result)
                self.assertEqual(result.cost, 7)
                self.assertEqual(len(result.path), 8)

    def test_problem2_given_example_and_free_goal(self):
        grid = Grid.from_rows(TERRAIN, marker_costs={"E": 0})
        for heuristic in ("zero", "goal_aware"):
            result = astar(grid, (0, 0), (4, 4), heuristic=heuristic)
            self.assert_valid_result(TERRAIN, (0, 0), (4, 4), result, {"E": 0})
            self.assertEqual(result.cost, 7)
        safe = goal_aware_heuristic(grid, (4, 4))
        self.assertEqual(safe((4, 3)), 0)
        self.assertEqual(reference_distances(TERRAIN, (4, 3), {"E": 0})[(4, 4)], 0)
        # Ordinary Manhattan distance is 1 here and therefore overestimates.
        self.assertGreater(abs(4 - 4) + abs(3 - 4), safe((4, 3)))

    def test_problem4_all_36_distances(self):
        grid = Grid.from_rows(TREASURE_MAP)
        expected = reference_matrix(TREASURE_MAP, TREASURE_POSITIONS)
        for i, start in enumerate(TREASURE_POSITIONS):
            for j, goal in enumerate(TREASURE_POSITIONS):
                result = astar(grid, start, goal)
                self.assert_valid_result(TREASURE_MAP, start, goal, result)
                self.assertEqual(result.cost, expected[i][j])
        self.assertEqual(expected, [list(row) for row in zip(*expected)])

    def test_disconnected_and_identical_endpoint(self):
        grid = Grid.from_rows([[1, "X", 1]])
        result = astar(grid, (0, 0), (0, 2))
        self.assertEqual(result.path, [])
        self.assertIsNone(result.cost)
        result = astar(grid, (0, 0), (0, 0), record_trace=True)
        self.assertEqual(result.path, [(0, 0)])
        self.assertEqual(result.cost, 0)
        self.assertEqual(result.stats.popped, 1)
        self.assertEqual(result.stats.expanded, 0)
        self.assertEqual(result.stats.discovered, 1)

    def test_metric_semantics_on_one_edge(self):
        result = astar(Grid.from_rows([[1, 1]]), (0, 0), (0, 1), record_trace=True)
        self.assertEqual(result.stats.popped, 2)
        self.assertEqual(result.stats.expanded, 1)
        self.assertEqual(result.stats.generated, 1)
        self.assertEqual(result.stats.discovered, 2)
        self.assertEqual(result.stats.reopened, 0)
        self.assertEqual(result.stats.stale_pops, 0)
        self.assertEqual(len(result.trace), result.stats.popped)
        self.assertEqual(sum(event["expanded"] for event in result.trace), result.stats.expanded)
        self.assertLessEqual(result.stats.preparation_ns, result.stats.elapsed_ns)

    def test_random_positive_and_zero_cost_grids(self):
        rng = random.Random(20261023)
        for case in range(48):
            height, width = rng.randint(2, 6), rng.randint(2, 6)
            minimum = case % 2
            rows = [["X" if rng.random() < .22 else rng.randint(minimum, 9)
                     for _ in range(width)] for _ in range(height)]
            start, goal = (0, 0), (height - 1, width - 1)
            rows[0][0], rows[-1][-1] = 1, 0 if minimum == 0 else 1
            grid = Grid.from_rows(rows)
            for heuristic in ("zero", "goal_aware"):
                for tie_break in ("fifo", "larger_g"):
                    with self.subTest(case=case, heuristic=heuristic, tie_break=tie_break):
                        result = astar(grid, start, goal, heuristic=heuristic,
                                       tie_break=tie_break)
                        self.assert_valid_result(rows, start, goal, result)
            if minimum == 1:
                self.assert_valid_result(rows, start, goal,
                                         astar(grid, start, goal, heuristic="manhattan"))

    def test_goal_aware_and_alt_are_admissible_and_consistent(self):
        rng = random.Random(98371)
        for case in range(12):
            rows = [["X" if rng.random() < .16 else rng.randint(case % 2, 6)
                     for _ in range(5)] for _ in range(4)]
            rows[0][0], rows[-1][-1] = 2, 0 if case % 2 == 0 else 1
            costs = parsed_costs(rows)
            grid = Grid.from_rows(rows)
            points = list(costs)
            landmarks = [points[0], points[len(points) // 2], points[-1]]
            index = LandmarkIndex.build(grid, landmarks)
            self.assertGreaterEqual(index.preprocessing_ns, 0)
            self.assertGreater(index.table_entries, 0)
            goal = (3, 4)
            for kind, heuristic in (("goal_aware", goal_aware_heuristic(grid, goal)),
                                    ("ALT", index.heuristic(goal))):
                self.assertEqual(heuristic(goal), 0)
                for position, _cost in costs.items():
                    with self.subTest(case=case, heuristic=kind, position=position):
                        exact = reference_distances(rows, position)[goal]
                        self.assertGreaterEqual(heuristic(position), 0)
                        self.assertLessEqual(heuristic(position), exact)
                        r, c = position
                        for adjacent in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if adjacent in costs:
                                self.assertLessEqual(heuristic(position),
                                                     costs[adjacent] + heuristic(adjacent))
                result = astar(grid, (0, 0), goal, heuristic=heuristic)
                self.assert_valid_result(rows, (0, 0), goal, result)

    def test_directed_entry_cost_and_alt(self):
        rows = [[9, 1, 2], [1, 3, 0]]
        grid = Grid.from_rows(rows)
        first = astar(grid, (0, 0), (1, 2)).cost
        reverse = astar(grid, (1, 2), (0, 0)).cost
        self.assertEqual(reverse - first, 9)
        index = LandmarkIndex.build(grid, [(0, 0), (1, 2)])
        for start in parsed_costs(rows):
            for goal in parsed_costs(rows):
                self.assert_valid_result(rows, start, goal,
                                         astar(grid, start, goal, heuristic=index.heuristic(goal)))

    def test_admissible_inconsistent_heuristic_reopens_closed_nodes(self):
        rows = [[8, 4, 1, 1], [2, 5, 3, 6], [9, 6, 3, 3], [6, 2, 5, 7]]
        estimates = [[21, 14, 16, 10], [17, 5, 5, 9], [11, 13, 8, 3], [2, 0, 1, 0]]
        heuristic = lambda p: estimates[p[0]][p[1]]
        for position in parsed_costs(rows):
            self.assertLessEqual(heuristic(position), reference_distances(rows, position)[(3, 3)])
        self.assertGreater(heuristic((1, 0)), rows[1][1] + heuristic((1, 1)))
        result = astar(Grid.from_rows(rows), (0, 0), (3, 3), heuristic=heuristic)
        self.assert_valid_result(rows, (0, 0), (3, 3), result)
        self.assertEqual(result.cost, 21)
        self.assertGreater(result.stats.reopened, 0)
        self.assertGreater(result.stats.stale_pops, 0)

    def test_bad_custom_heuristics_are_rejected(self):
        grid = Grid.from_rows([[1, 1]])
        for invalid in (-1, float("nan"), float("inf"), True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                astar(grid, (0, 0), (0, 1), heuristic=lambda _pos: invalid)
        with self.assertRaises(ValueError):
            astar(grid, (0, 0), (0, 1), heuristic=lambda _pos: 1)

    def test_internal_zero_cost_makes_raw_manhattan_suboptimal(self):
        # Only one zero-cost interior cell is enough to break the raw heuristic.
        # This is an extension case, not a claim that the PDF example is wrong.
        rows = [["S", 3, "E"], [1, 0, 1]]
        grid = Grid.from_rows(rows, marker_costs={"E": 0})
        for tie_break in ("fifo", "larger_g"):
            raw = astar(grid, (0, 0), (0, 2), heuristic="manhattan", tie_break=tie_break)
            self.assertEqual(raw.cost, 3)
            for heuristic in ("zero", "goal_aware"):
                safe = astar(grid, (0, 0), (0, 2), heuristic=heuristic, tie_break=tie_break)
                self.assert_valid_result(rows, (0, 0), (0, 2), safe, {"E": 0})
                self.assertEqual(safe.cost, 2)

    def test_invalid_grids_and_endpoints_are_rejected(self):
        for rows in ([], [[]], [[1], [1, 2]], [[-1]], [[1.5]], [[True]], [["unknown"]]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                Grid.from_rows(rows)
        grid = Grid.from_rows([[1, "X"], [1, 1]])
        for start, goal in (((0, 1), (1, 1)), ((0, 0), (3, 1)), ((-1, 0), (1, 1))):
            with self.assertRaises((ValueError, TypeError)):
                astar(grid, start, goal)
        for kwargs in ({"heuristic": "unknown"}, {"tie_break": "unknown"}):
            with self.assertRaises((ValueError, TypeError)):
                astar(grid, (0, 0), (1, 1), **kwargs)


if __name__ == "__main__":
    unittest.main()
