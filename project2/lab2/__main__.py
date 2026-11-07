"""命令行输入输出保持简洁；详细数据通过 experiments 子命令输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="实验二：A*、Minimax、Alpha-beta")
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", help="求解一题；默认读取文档样例")
    solve.add_argument("problem", type=int, choices=(1, 2, 3, 4))
    solve.add_argument("--input", type=Path, help="自定义 JSON 输入")
    solve.add_argument("--depth", type=int, default=4)
    solve.add_argument("--algorithm", choices=("minimax", "alpha_beta"), default="alpha_beta")
    samples = sub.add_parser("samples", help="运行并验证全部文档样例")
    samples.add_argument("--output", type=Path, default=Path("results/samples.json"))
    exp = sub.add_parser("experiments", help="运行对照与拓展实验，保存原始数据和图表")
    exp.add_argument("--output", type=Path, default=Path("results"))
    exp.add_argument("--repeats", type=int, default=15)
    exp.add_argument("--warmups", type=int, default=3)
    exp.add_argument("--seed", type=int, default=20261023)
    exp.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.command == "solve":
        from .tasks import load_sample, solve_maze, solve_weighted, solve_game_tree, solve_treasures
        data = json.loads(args.input.read_text(encoding="utf-8")) if args.input else load_sample(args.problem)
        if args.problem == 1:
            # Python tuple 列表精确遵循讲义的展示格式。
            print(solve_maze(data["start"], data["goal"], data["grid"]))
        elif args.problem == 2:
            print(solve_weighted(data["start"], data["goal"], data["grid"]))
        elif args.problem == 3:
            print(solve_game_tree(data))
        else:
            print(json.dumps(solve_treasures(data, args.depth, args.algorithm), ensure_ascii=False, indent=2))
    elif args.command == "samples":
        from .experiments import run_samples
        result = run_samples(args.output.parent)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
        print(f"全部课例验证通过，详细结果：{args.output}")
    else:
        if args.repeats < 3 or args.warmups < 0:
            parser.error("repeats 至少为 3，warmups 不可为负")
        from .experiments import run_experiments
        run_experiments(args.output, args.repeats, args.warmups, args.seed, not args.no_plots)


if __name__ == "__main__":
    main()
