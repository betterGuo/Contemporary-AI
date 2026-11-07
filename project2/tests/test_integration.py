"""Exercise the public task wrappers, CLI, sample exports and small benchmarks.

Only temporary output folders are used. The full expansion benchmark suite is
intentionally excluded so the normal correctness suite remains lightweight.
"""

import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from lab2.experiments import Benchmarks, benchmark_samples, run_samples
from lab2.tasks import ROOT, load_sample, solve_maze, solve_weighted
from tests.oracles import (
    TREASURE_MAP, TREASURE_POSITIONS, TREASURE_VALUES,
    reference_matrix, reference_treasure,
)


class IntegrationTests(unittest.TestCase):
    def run_cli(self, *arguments, check=True):
        return subprocess.run([sys.executable, "-m", "lab2", *map(str, arguments)],
                              cwd=ROOT, capture_output=True, text=True, check=check,
                              timeout=15)

    def test_four_cli_problem_outputs(self):
        first = ast.literal_eval(self.run_cli("solve", 1).stdout)
        self.assertEqual(first[0], (0, 0))
        self.assertEqual(first[-1], (3, 4))
        self.assertEqual(len(first), 8)
        self.assertEqual(self.run_cli("solve", 2).stdout.strip(), "7")
        self.assertEqual(self.run_cli("solve", 3).stdout.strip(), "5")
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            response = json.loads(self.run_cli("solve", 4, "--depth", depth,
                                               "--algorithm", "minimax").stdout)
            self.assertEqual(response["best_target"], target)
            self.assertEqual(response["best_value"], value)
            self.assertEqual(response["distance_matrix"], reference_matrix(TREASURE_MAP, TREASURE_POSITIONS))

    def test_custom_cli_json_and_wrapper_no_path_contracts(self):
        blocked = {"start": [0, 0], "goal": [0, 2], "grid": [["S", "X", "E"]]}
        self.assertEqual(solve_maze(**blocked), [])
        self.assertEqual(solve_weighted(**blocked), -1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            path.write_text(json.dumps(blocked), encoding="utf-8")
            self.assertEqual(self.run_cli("solve", 1, "--input", path).stdout.strip(), "[]")
            self.assertEqual(self.run_cli("solve", 2, "--input", path).stdout.strip(), "-1")
        self.assertEqual(solve_weighted((0, 0), (0, 1), [["S", "E"]]), 0)

    def test_samples_exports_and_traces_match_independent_results(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_samples(output)
            self.assertTrue(result["all_passed"])
            self.assertEqual(json.loads((output / "samples.json").read_text()),
                             json.loads(json.dumps(result)))
            self.assertEqual(result["problem2"]["accumulation"][-1]["entering_cost"], 0)
            self.assertEqual(result["problem2"]["accumulation"][-1]["cumulative_cost"], 7)
            matrix = reference_matrix(TREASURE_MAP, TREASURE_POSITIONS)
            self.assertEqual(result["problem4"]["distance_matrix"], matrix)
            for depth in (2, 4):
                values = result["problem4"]["exact_root_action_values"][str(depth)]
                for target in range(2, 6):
                    gain = TREASURE_VALUES[target - 2] - matrix[0][target]
                    exact, _ = reference_treasure(
                        matrix, TREASURE_VALUES, depth - 1, pos_a=target,
                        remaining=tuple(t for t in range(2, 6) if t != target),
                        score_a=gain, player="B")
                    self.assertEqual(values[f"T{target - 1}"], exact)
                final = result["problem4"]["optimal_lines"][str(depth)][-1]
                self.assertEqual(final["utility"], -1 if depth == 2 else 3)
            with (output / "distance_searches.csv").open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 36)
            self.assertEqual(len(list((output / "traces").glob("*.json"))), 7)
            for problem in (1, 2):
                events = json.loads((output / "traces" / f"problem{problem}.json").read_text())
                self.assertEqual(len(events), result[f"problem{problem}"]["stats"]["popped"])
            self.assertTrue(all(json.loads(path.read_text()) for path in (output / "traces").glob("*.json")))

    def test_samples_cli_writes_requested_file(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "answers.json"
            response = self.run_cli("samples", "--output", target)
            self.assertIn("全部课例验证通过", response.stdout)
            result = json.loads(target.read_text(encoding="utf-8"))
            self.assertTrue(result["all_passed"])
            self.assertEqual(result["answers"]["problem3_value"], 5)

    def test_required_benchmark_rows_preserve_expected_values(self):
        # Just 60 small configurations; no generated large maps or scaling cases.
        benchmark = Benchmarks(repeats=3, warmups=1, seed=20261023)
        benchmark_samples(benchmark)
        self.assertEqual(len(benchmark.summary), 60)
        self.assertEqual(len(benchmark.raw), 180)
        matrix = reference_matrix(TREASURE_MAP, TREASURE_POSITIONS)
        for row in benchmark.summary:
            self.assertEqual(row["repeats"], 3)
            self.assertLessEqual(row["min_us"], row["p25_us"])
            self.assertLessEqual(row["p25_us"], row["median_us"])
            self.assertLessEqual(row["median_us"], row["p75_us"])
            if row["case"] in {"problem1", "problem2"}:
                self.assertEqual(row["cost"], 7)
            elif row["case"] == "problem3":
                self.assertEqual(row["value"], 5)
            else:
                exact, optimal_indices = reference_treasure(matrix, TREASURE_VALUES, row["depth"])
                self.assertEqual(row["value"], exact)
                self.assertIn(row["best_target"], {f"T{i - 1}" for i in optimal_indices})
        for row in benchmark.raw:
            self.assertGreaterEqual(row["wall_ns"], row["algorithm_elapsed_ns"])

    def test_cli_rejects_invalid_benchmark_settings_before_running(self):
        for arguments in (("--repeats", "2"), ("--warmups", "-1")):
            response = self.run_cli("experiments", *arguments, check=False)
            self.assertEqual(response.returncode, 2)
            self.assertIn("repeats 至少为 3", response.stderr)


if __name__ == "__main__":
    unittest.main()
