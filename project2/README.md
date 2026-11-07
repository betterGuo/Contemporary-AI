# 当代人工智能 · 实验二：搜索算法

按照《实验二要求_new.pdf》（2026-10-15，17页）实现四题，保留给定测试、独立正确性测试、搜索轨迹、性能对照和创新拓展数据。代码使用 Python 标准库；绘图另需 Matplotlib。

## 快速运行

在本文件所在目录执行，推荐 Python 3.12（语法要求 Python 3.10+）：

```bash
# 四道题的原始样例与预期值逐项断言
python3 -m lab2 samples

# 独立参考算法、随机差分、边界条件及命令行集成测试
python3 -m unittest discover -s tests -v

# 本次提交数据的完整复现命令：3轮预热、21轮计时
python3 -m lab2 experiments --repeats 21 --warmups 3 --seed 20261023

# 没有 Matplotlib 时，仍可完成全部求解、测试和数据采集
python3 -m lab2 experiments --repeats 21 --warmups 3 --no-plots
```

若缺少绘图库，可在自己的虚拟环境执行 `python3 -m pip install -r requirements.txt`。当前验证环境为 Python 3.12.6 / Matplotlib 3.10.0 / macOS arm64，机器信息与源文件 SHA-256 见 `results/manifest.json`。运行时间会随机器与负载变化；相同输入、算法和访问规则的结果及计数可复现。

单题入口：

```bash
python3 -m lab2 solve 1
python3 -m lab2 solve 2
python3 -m lab2 solve 3
python3 -m lab2 solve 4 --depth 2 --algorithm minimax
python3 -m lab2 solve 4 --depth 4 --algorithm alpha_beta
python3 -m lab2 solve 2 --input data/problem2.json
```

`solve 1` 输出 Python 坐标元组列表，`solve 2/3` 输出整数，`solve 4` 输出含距离矩阵、最优首步和效用的 JSON。自定义 JSON 格式参照 `data/`。每次运行会更新选定输出位置。保留历史运行时，分别使用 `samples --output results/run_name/samples.json`（文件路径）和 `experiments --output results/run_name`（目录路径）。

## 文件与阅读顺序

| 路径 | 用途 |
|---|---|
| `lab2/pathfinding.py` | A*、四连通加权网格、终点感知启发函数、有向 ALT 地标索引 |
| `lab2/games.py` | Minimax、Alpha-beta、宝藏状态、收益排序、带界类型的缓存 |
| `lab2/tasks.py` | 对应四道题的简洁函数接口及36次 A* 距离矩阵计算 |
| `lab2/experiments.py` | 重复计时、对照实验、输入生成、轨迹和 CSV/JSON 导出 |
| `lab2/plots.py` | 从保存的数据生成报告用 PNG/SVG 图表 |
| `data/problem1.json` 至 `problem4.json` | 手工核对 PDF 后录入的全部原题输入 |
| `tests/` | 独立 Dijkstra/穷举参考实现及随机、边界和集成测试 |
| `docs/ALGORITHMS.md` | 状态设计、算法推导、启发可采纳性、缓存正确性和研究出处 |
| `docs/EXPERIMENTS.md` | 实验设置、统计字段、对照设计和评分点对应 |
| `results/FINDINGS.md` | **优先阅读：由真实实验数据生成的结果表及分析提醒** |
| `results/summary.csv` / `summary.json` | 每种配置的节点数、中位耗时和四分位数 |
| `results/raw_runs.csv` | 每次独立计时的原始记录，保留纳秒精度 |
| `results/figures/` | 可直接用于报告的高分辨率 PNG 和矢量 SVG |
| `results/traces/` | 前三题与题四两个深度两算法的逐步搜索轨迹 |
| `results/generated/` | 自设地图、查询、宝藏布局、价值和参考答案 |

`results/samples.json` 保存路径、逐步累计代价、根动作精确效用和最优对局过程；`distance_matrix.csv` 保存题四6×6矩阵；`distance_searches.csv` 保存36个有序点对的单次 A* 搜索统计。

## 原题结果与规则

| 题目 | 文档期望 | 默认算法/行为 |
|---|---|---|
| 题一 | 7步最短路径，8个坐标 | A*、四方向、单位代价；同长路径任意一条；无解 `[]` |
| 题二 | 最小代价7 | 进入格子计费，起点不收费，E进入代价0；无解 `-1` |
| 题三 | 根 MAX 得分5 | Alpha-beta、严格左到右；逆序仅作为对照 |
| 题四 H=2 | 首步T2，U=-1 | Minimax与Alpha-beta分别计算 |
| 题四 H=4 | 首步T3，U=3 | Minimax与Alpha-beta分别计算 |

题四宝藏价值为 T1=9、T2=6、T3=7、T4=7；A/B/T 所在格进入代价均为1。行动顺序 A→B→A→B，H按**一次玩家行动**计数。每次只领取选中的宝藏，经过其他宝藏不收取；已取位置仍可通行，另一玩家不阻挡。每次收益为 `价值−移动代价`，负收益动作仍合法；效用为 `Score_A−Score_B`。

原题距离矩阵的行列顺序均为 A、B、T1、T2、T3、T4：

```text
     A   B  T1  T2  T3  T4
A    0  12  10   4   8   8
B   12   0   8   8   4  10
T1  10   8   0   6   6  12
T2   4   8   6   0   6   6
T3   8   4   6   6   0   6
T4   8  10  12   6   6   0
```

一般“进入格子计费”的图是有向加权图。本题特殊标记权重均相同，矩阵恰好对称；实现仍逐个有序点对计算，不将一般的对称性当作假设。

## 创新与拓展

1. **终点感知安全启发式**：针对E=0修正曼哈顿下界；审计每个格子的真实剩余代价。额外内部零权反例展示原始曼哈顿代价3、安全启发代价2。仅E为零的特殊条件下，原始曼哈顿虽不可采纳却仍可证明返回最优，详见算法文档，避免错误归因。
2. **A*同分规则消融**：只在相同f时比较FIFO与larger-g，覆盖开放网格、绕墙、加权和内部零权地图。
3. **有向ALT地标启发**：正向与反向距离共同给出下界，正确处理“进入格子”的方向差异。计入预处理、启发工厂和查询成本；除每图12查询外，连续测量Q=1/12/60/120的真实批次，检验摊销条件。
4. **收益排序与安全缓存**：即时收益排序只改变访问顺序；缓存键包含位置、剩余宝藏、玩家和深度，值规范化为未来效用增量，并区分EXACT/LOWER/UPPER。扩展到4/6/8个宝藏，8宝藏使用三个种子，比较完整消融数据。

这些是经典方法的适配、组合与验证。保留无收益甚至更慢的配置，分析适用条件；不以“更复杂”替代优化证据。

## 输入边界与复现约定

- 坐标从0开始，顺序为 `(row, col)`；寻路邻居固定为上、右、下、左，同分再按入堆序号保持稳定。
- 通用网格支持非负整数权重、数字字符串、`X`/`None`障碍和题目标记。负权、空/非矩形地图、非法坐标、起终点为障碍均明确抛出 `ValueError`。无路径与无效输入分开处理。
- 通用 `astar` 无解时 `cost=None`，四题薄接口再转换为题目要求。`start==goal` 返回一个坐标，代价0。
- 题二 `S` 若被再次进入按1计费，这是讲义未定义的输入扩展约定；起点初始不收费，不影响非负权最优路径。E按0处理。
- 博弈在深度耗尽或宝藏取完时评价当前U；自设非连通场景中，当前玩家无可达宝藏时直接终局，不另加未定义的跳过规则。给定样例全部可达。
- 所有计时搜索关闭trace，博弈缓存每次调用新建。读取文件、绘图、结果序列化不进入计时。内核计时与包含入口校验的外层计时同时记录。

文档截止时间为 **2026年11月12日23:59:59**，提交方式待课程群通知。讲义未规定报告文件格式或打包命名；当前产物为代码与报告素材，正式实验报告可基于已有结果继续编写。
