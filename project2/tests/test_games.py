"""Exhaustive independent references for alpha-beta and treasure allocation."""

import random
import unittest

from lab2.games import GameState, TreeNode, TreasureGame, solve_tree, solve_treasure, tree_from_dict
from tests.oracles import (
    TREASURE_MAP, TREASURE_NAMES, TREASURE_POSITIONS, TREASURE_VALUES,
    reference_matrix, reference_tree, reference_treasure,
)


def make_game(matrix, values):
    names = ["A", "B"] + [f"T{i + 1}" for i in range(len(values))]
    return TreasureGame(matrix, names, dict(zip(names[2:], values))), names


def example_tree():
    max1 = TreeNode(children=[TreeNode(value=3), TreeNode(value=5)])
    max2 = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)])
    max3 = TreeNode(children=[TreeNode(value=1), TreeNode(value=2)])
    return TreeNode(children=[TreeNode(children=[max1, max2]),
                              TreeNode(children=[max2, max3])])


class TreeTests(unittest.TestCase):
    def test_problem3_given_shared_subtree_example(self):
        root = example_tree()
        self.assertEqual(reference_tree(root), 5)
        for algorithm in ("minimax", "alpha_beta"):
            for order in ("natural", "reverse"):
                with self.subTest(algorithm=algorithm, order=order):
                    result = solve_tree(root, algorithm=algorithm, order=order, record_trace=True)
                    self.assertEqual(result.value, 5)
                    self.assertGreater(result.stats.visited, 0)
                    self.assertGreaterEqual(result.stats.elapsed_ns, 0)
        # This is a tree traversal, so sharing Python objects must not remove visits.
        full = solve_tree(root, algorithm="minimax")
        self.assertEqual(full.stats.visited, 15)
        self.assertEqual(full.stats.leaves, 8)

    def test_tree_dict_roundtrip(self):
        root = tree_from_dict({
            "is_maximizing_player": True,
            "children": [
                {"is_maximizing_player": False,
                 "children": [{"value": -2}, {"value": 4}]},
                {"is_maximizing_player": False,
                 "children": [{"value": 0}, {"value": 3}]},
            ],
        })
        self.assertEqual(solve_tree(root).value, 0)

    def test_true_cutoffs_and_direct_branch_metrics(self):
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(value=3), TreeNode(value=5)]),
            TreeNode(children=[TreeNode(value=1), TreeNode(value=2)]),
        ])
        result = solve_tree(root)
        self.assertEqual(result.value, 3)
        self.assertEqual(result.stats.visited, 6)
        self.assertEqual(result.stats.expanded, 3)
        self.assertEqual(result.stats.leaves, 3)
        self.assertEqual(result.stats.cutoffs, 1)
        self.assertEqual(result.stats.pruned_branches, 1)
        full = solve_tree(root, algorithm="minimax")
        self.assertEqual(full.stats.visited, 7)
        self.assertEqual(full.stats.cutoffs, 0)
        self.assertEqual(full.stats.pruned_branches, 0)
        # A single skipped subtree counts as one direct branch, not three nodes.
        root.children[1].children[1] = TreeNode(children=[TreeNode(value=10), TreeNode(value=20)])
        result = solve_tree(root)
        self.assertEqual(result.stats.pruned_branches, 1)
        self.assertEqual(solve_tree(root, algorithm="minimax").stats.visited - result.stats.visited, 3)

    def test_bound_crossing_at_last_child_is_not_a_cutoff(self):
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(value=3), TreeNode(value=5)]),
            TreeNode(children=[TreeNode(value=4), TreeNode(value=2)]),
        ])
        result = solve_tree(root)
        self.assertEqual(result.value, 3)
        self.assertEqual(result.stats.visited, 7)
        self.assertEqual(result.stats.cutoffs, 0)
        self.assertEqual(result.stats.pruned_branches, 0)

    def test_random_trees_against_exhaustive_reference(self):
        rng = random.Random(18710)

        def build(depth):
            if not depth or rng.random() < .16:
                return TreeNode(value=rng.randint(-25, 25))
            return TreeNode(children=[build(depth - 1) for _ in range(rng.randint(1, 4))])

        for case in range(40):
            root = build(rng.randint(1, 5))
            exact = reference_tree(root)
            full = solve_tree(root, algorithm="minimax")
            self.assertEqual(full.value, exact)
            for order in ("natural", "reverse"):
                with self.subTest(case=case, order=order):
                    result = solve_tree(root, order=order)
                    self.assertEqual(result.value, exact)
                    self.assertLessEqual(result.stats.visited, full.stats.visited)
                    self.assertEqual(result.stats.expanded + result.stats.leaves, result.stats.visited)

    def test_leaf_and_invalid_empty_tree(self):
        result = solve_tree(TreeNode(value=-7))
        self.assertEqual(result.value, -7)
        self.assertEqual(result.stats.visited, 1)
        self.assertEqual(result.stats.leaves, 1)
        self.assertEqual(result.stats.expanded, 0)
        for kwargs in ({"algorithm": "unknown"}, {"order": "unknown"}):
            with self.assertRaises((ValueError, TypeError)):
                solve_tree(TreeNode(value=1), **kwargs)
        with self.assertRaises((ValueError, TypeError)):
            solve_tree(TreeNode())

    def test_cyclic_or_inconsistent_trees_are_rejected(self):
        root = TreeNode()
        root.children.append(root)
        with self.assertRaises(ValueError):
            solve_tree(root)
        for tree in ({"is_maximizing_player": False, "value": 3},
                     {"value": 3, "children": [{"value": 2}]},
                     {"children": [{"value": 2.5}]}):
            with self.subTest(tree=tree), self.assertRaises((ValueError, TypeError)):
                tree_from_dict(tree)


class TreasureTests(unittest.TestCase):
    def assert_matches_reference(self, matrix, values, depth, *, state=None):
        game, names = make_game(matrix, values)
        kwargs = {}
        if state is not None:
            kwargs = {"pos_a": names.index(state.pos_a), "pos_b": names.index(state.pos_b),
                      "remaining": tuple(names.index(t) for t in state.remaining),
                      "score_a": state.score_a, "score_b": state.score_b, "player": state.player}
        exact, optimal_indices = reference_treasure(matrix, values, depth, **kwargs)
        optimal_names = {names[index] for index in optimal_indices}
        for algorithm, cache in (("minimax", False), ("minimax", True),
                                 ("alpha_beta", False), ("alpha_beta", True)):
            for order in ("natural", "reverse", "greedy"):
                with self.subTest(depth=depth, algorithm=algorithm, cache=cache, order=order):
                    result = solve_treasure(game, depth, algorithm=algorithm, order=order,
                                            use_cache=cache, state=state)
                    self.assertEqual(result.value, exact)
                    if optimal_names:
                        self.assertIn(result.best_target, optimal_names)
                    else:
                        self.assertIsNone(result.best_target)
                    self.assertGreaterEqual(result.stats.visited, 1)
                    self.assertGreaterEqual(result.stats.elapsed_ns, 0)
        return exact, optimal_names

    def test_problem4_depth2_and_depth4_examples(self):
        matrix = reference_matrix(TREASURE_MAP, TREASURE_POSITIONS)
        for depth, expected_value, expected_target in ((2, -1, "T2"), (4, 3, "T3")):
            value, targets = self.assert_matches_reference(matrix, TREASURE_VALUES, depth)
            self.assertEqual(value, expected_value)
            self.assertEqual(targets, {expected_target})

    def test_successor_updates_only_acting_player_and_selected_treasure(self):
        matrix = reference_matrix(TREASURE_MAP, TREASURE_POSITIONS)
        game, _ = make_game(matrix, TREASURE_VALUES)
        initial = game.initial_state()
        after_a = game.successor(initial, "T3")
        self.assertEqual(after_a.pos_a, "T3")
        self.assertEqual(after_a.pos_b, "B")
        self.assertEqual(after_a.score_a, 7 - matrix[0][4])
        self.assertEqual(after_a.score_b, 0)
        self.assertEqual(after_a.player, "B")
        self.assertEqual(set(after_a.remaining), {"T1", "T2", "T4"})
        self.assertEqual(initial.remaining, ("T1", "T2", "T3", "T4"))
        after_b = game.successor(after_a, "T4")
        self.assertEqual(after_b.pos_a, "T3")
        self.assertEqual(after_b.pos_b, "T4")
        self.assertEqual(after_b.score_a, after_a.score_a)
        self.assertEqual(after_b.score_b, 7 - matrix[1][5])
        self.assertEqual(after_b.player, "A")
        with self.assertRaises(ValueError):
            game.successor(after_b, "T4")

    def test_random_directed_games_multiple_depths(self):
        rng = random.Random(20261024)
        for case in range(14):
            count = rng.randint(2, 5)
            size = count + 2
            matrix = [[0 if i == j else rng.randint(0, 15)
                       for j in range(size)] for i in range(size)]
            values = [rng.randint(1, 15) for _ in range(count)]
            for depth in {0, 1, min(2, count), count}:
                with self.subTest(case=case):
                    self.assert_matches_reference(matrix, values, depth)

    def test_nonzero_scores_current_player_and_partial_state(self):
        rng = random.Random(106)
        count = 5
        matrix = [[0 if i == j else rng.randint(1, 12)
                   for j in range(count + 2)] for i in range(count + 2)]
        values = [4, 6, 9, 5, 8]
        for player in ("A", "B"):
            for score_a, score_b in ((17, -9), (-13, 21), (0, 0)):
                state = GameState("T1", "T2", ("T3", "T4", "T5"),
                                  score_a=score_a, score_b=score_b, player=player)
                for depth in (1, 3):
                    self.assert_matches_reference(matrix, values, depth, state=state)

    def test_normalized_transpositions_have_real_cache_hits(self):
        # Different earlier assignments produce the same positions and remaining
        # treasures with different accumulated U. Caching total U would be wrong.
        rng = random.Random(777)
        count = 6
        matrix = [[0 if i == j else rng.randint(1, 18)
                   for j in range(count + 2)] for i in range(count + 2)]
        values = [2, 8, 4, 14, 5, 11]
        exact, _ = self.assert_matches_reference(matrix, values, count)
        game, _ = make_game(matrix, values)
        plain = solve_treasure(game, count, algorithm="minimax", use_cache=False)
        cached = solve_treasure(game, count, algorithm="minimax", use_cache=True)
        self.assertEqual(cached.value, exact)
        self.assertGreater(cached.stats.cache_hits, 0)
        self.assertGreater(cached.stats.cache_stores, 0)
        self.assertLess(cached.stats.visited, plain.stats.visited)

    def test_negative_gain_must_still_select_a_treasure(self):
        matrix = [[0, 1, 10], [1, 0, 12], [10, 12, 0]]
        value, targets = self.assert_matches_reference(matrix, [3], 1)
        self.assertEqual(value, -7)
        self.assertEqual(targets, {"T1"})

    def test_all_equal_optima_and_depth_beyond_terminal(self):
        matrix = [[0 for _ in range(5)] for _ in range(5)]
        value, targets = self.assert_matches_reference(matrix, [4, 4, 4], 8)
        self.assertEqual(value, 4)
        self.assertEqual(targets, {"T1", "T2", "T3"})

    def test_unreachable_actions_and_no_legal_move(self):
        matrix = [[0, 1, None], [1, 0, 2], [None, 2, 0]]
        game, _ = make_game(matrix, [8])
        result = solve_treasure(game, 2)
        self.assertEqual(result.value, 0)
        self.assertIsNone(result.best_target)
        self.assertEqual(result.stats.leaves, 1)
        state = GameState("A", "B", ("T1",), player="B")
        result = solve_treasure(game, 1, state=state)
        self.assertEqual(result.value, -6)
        self.assertEqual(result.best_target, "T1")

    def test_invalid_parameters_are_rejected(self):
        good = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        for matrix in ([[0, 1]], [[0, -1, 2], [1, 0, 3], [2, 3, 0]]):
            with self.assertRaises((ValueError, TypeError)):
                make_game(matrix, [4])
        with self.assertRaises((ValueError, TypeError)):
            TreasureGame(good, ["A", "B", "T1"], {})
        game, _ = make_game(good, [4])
        for depth in (-1, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                solve_treasure(game, depth)
        for kwargs in ({"algorithm": "unknown"}, {"order": "unknown"}):
            with self.assertRaises((ValueError, TypeError)):
                solve_treasure(game, 1, **kwargs)


if __name__ == "__main__":
    unittest.main()
