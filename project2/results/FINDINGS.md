# 实验数据摘要（由程序生成）

本文件供后续实验报告使用。耗时为本机重复执行的中位数；完整原始样本保存在 raw_runs.csv。

## 文档样例与基础算法

所有给定样例已通过。以下使用默认访问顺序、关闭博弈缓存。

| 题目/深度 | 算法 | 结果 | 有效出堆/递归访问 | 展开 | 叶评价 | 剪枝次数 | 直接跳过分支 | 中位耗时 μs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| problem1 | astar | 7 | 14 | 13 | - | - | - | 32.08 |
| problem2 | astar | 7 | 11 | 10 | - | - | - | 34.79 |
| problem3 | minimax | 5 | 15 | 7 | 8 | 0 | 0 | 8.58 |
| problem3 | alpha_beta | 5 | 14 | 7 | 7 | 1 | 1 | 9.83 |
| problem4 H=2 | minimax | T2 / -1 | 17 | 5 | 12 | 0 | 0 | 40.58 |
| problem4 H=2 | alpha_beta | T2 / -1 | 13 | 5 | 8 | 2 | 4 | 34.04 |
| problem4 H=4 | minimax | T3 / 3 | 65 | 41 | 24 | 0 | 0 | 158.54 |
| problem4 H=4 | alpha_beta | T3 / 3 | 47 | 31 | 16 | 5 | 6 | 123.50 |

## 题四首步的精确效用

以下每个动作都独立全窗口搜索，不能从被剪枝的内部返回值直接推定。

| 深度 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| 2 | -4 | -1 | -2 | -4 |
| 4 | -5 | 1 | 3 | -7 |

## 创新与拓展数据

### A* 同分规则与 ALT

| 场景 | 策略 | 12次查询展开合计 | 在线中位总耗时 ms | 含预处理中位总耗时 ms |
|---|---|---:|---:|---:|
| open_20261023 | dijkstra | 5381 | 15.294 | 15.294 |
| open_20261023 | goal_aware_fifo | 1864 | 6.075 | 6.075 |
| open_20261023 | goal_aware_larger_g | 278 | 1.452 | 1.452 |
| open_20261023 | alt_larger_g | 278 | 2.386 | 14.334 |
| walls_20261023 | dijkstra | 2158 | 5.610 | 5.610 |
| walls_20261023 | goal_aware_fifo | 1696 | 4.980 | 4.980 |
| walls_20261023 | goal_aware_larger_g | 1501 | 4.585 | 4.585 |
| walls_20261023 | alt_larger_g | 517 | 3.244 | 11.354 |
| weighted_20261023 | dijkstra | 2415 | 6.389 | 6.389 |
| weighted_20261023 | goal_aware_fifo | 785 | 2.612 | 2.612 |
| weighted_20261023 | goal_aware_larger_g | 652 | 2.264 | 2.264 |
| weighted_20261023 | alt_larger_g | 324 | 2.013 | 10.908 |
| weighted_20261024 | dijkstra | 2635 | 7.279 | 7.279 |
| weighted_20261024 | goal_aware_fifo | 990 | 3.353 | 3.353 |
| weighted_20261024 | goal_aware_larger_g | 872 | 3.033 | 3.033 |
| weighted_20261024 | alt_larger_g | 476 | 2.796 | 11.758 |
| weighted_20261025 | dijkstra | 3000 | 8.092 | 8.092 |
| weighted_20261025 | goal_aware_fifo | 1179 | 3.839 | 3.839 |
| weighted_20261025 | goal_aware_larger_g | 1030 | 3.420 | 3.420 |
| weighted_20261025 | alt_larger_g | 553 | 3.144 | 12.386 |
| zero_cost_20261023 | dijkstra | 2201 | 6.011 | 6.011 |
| zero_cost_20261023 | goal_aware_fifo | 2145 | 6.415 | 6.415 |
| zero_cost_20261023 | goal_aware_larger_g | 2134 | 6.440 | 6.440 |
| zero_cost_20261023 | alt_larger_g | 316 | 2.050 | 11.221 |
| zero_cost_20261024 | dijkstra | 2636 | 7.157 | 7.157 |
| zero_cost_20261024 | goal_aware_fifo | 2372 | 7.053 | 7.053 |
| zero_cost_20261024 | goal_aware_larger_g | 2348 | 7.015 | 7.015 |
| zero_cost_20261024 | alt_larger_g | 363 | 2.328 | 11.456 |

ALT 同时报告正反向地标距离表的预处理耗时、展开数和表项数。四个地标取接近四角的可达格；
每图第一个查询使用对角角点（终点也是地标），其余11个查询由固定种子采样，避免只展示有利特例。
同一trial内的12个在线查询耗时相加，另加一次预处理，属于组成项成本核算，并非连续批次墙钟测量。

### 连续实际查询批次（每批重新预处理）

此组的120个查询全部随机采样，取相同列表的前Q个，ALT每批新建一次索引。总耗时直接测量，含预处理。

| 场景 | Q | 安全曼哈顿+larger-g 总耗时 ms | ALT+larger-g 总耗时 ms |
|---|---:|---:|---:|
| walls_20261023 | 1 | 0.650 | 8.504 |
| walls_20261023 | 12 | 6.827 | 12.743 |
| walls_20261023 | 60 | 31.775 | 28.117 |
| walls_20261023 | 120 | 64.881 | 48.604 |
| weighted_20261023 | 1 | 0.089 | 8.988 |
| weighted_20261023 | 12 | 3.351 | 11.724 |
| weighted_20261023 | 60 | 18.546 | 25.402 |
| weighted_20261023 | 120 | 38.954 | 42.393 |
| zero_cost_20261023 | 1 | 0.189 | 9.329 |
| zero_cost_20261023 | 12 | 5.842 | 11.489 |
| zero_cost_20261023 | 60 | 37.848 | 25.040 |
| zero_cost_20261023 | 120 | 74.754 | 40.249 |

### 宝藏数量扩展与缓存

| 场景 | 策略 | 访问 | 剪枝 | 跳过直接分支 | 缓存命中 | 访问减少 | 中位耗时 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| treasures_4_seed_20261023 | minimax/natural/cache=False | 65 | 0 | 0 | 0 | 0.0% | 0.165 |
| treasures_4_seed_20261023 | minimax/natural/cache=True | 65 | 0 | 0 | 12 | 0.0% | 0.184 |
| treasures_4_seed_20261023 | alpha_beta/natural/cache=False | 41 | 4 | 6 | 0 | 36.9% | 0.114 |
| treasures_4_seed_20261023 | alpha_beta/reverse/cache=False | 49 | 4 | 5 | 0 | 24.6% | 0.135 |
| treasures_4_seed_20261023 | alpha_beta/greedy/cache=False | 31 | 5 | 8 | 0 | 52.3% | 0.093 |
| treasures_4_seed_20261023 | alpha_beta/greedy/cache=True | 31 | 5 | 8 | 4 | 52.3% | 0.104 |
| treasures_6_seed_20261023 | minimax/natural/cache=False | 1957 | 0 | 0 | 0 | 0.0% | 4.897 |
| treasures_6_seed_20261023 | minimax/natural/cache=True | 997 | 0 | 0 | 510 | 49.1% | 2.731 |
| treasures_6_seed_20261023 | alpha_beta/natural/cache=False | 371 | 66 | 106 | 0 | 81.0% | 1.018 |
| treasures_6_seed_20261023 | alpha_beta/reverse/cache=False | 832 | 68 | 107 | 0 | 57.5% | 2.238 |
| treasures_6_seed_20261023 | alpha_beta/greedy/cache=False | 269 | 64 | 113 | 0 | 86.3% | 0.823 |
| treasures_6_seed_20261023 | alpha_beta/greedy/cache=True | 219 | 62 | 111 | 75 | 88.8% | 0.757 |
| treasures_8_seed_20261023 | minimax/natural/cache=False | 109601 | 0 | 0 | 0 | 0.0% | 283.501 |
| treasures_8_seed_20261023 | minimax/natural/cache=True | 10817 | 0 | 0 | 7224 | 90.1% | 30.232 |
| treasures_8_seed_20261023 | alpha_beta/natural/cache=False | 7261 | 1286 | 2184 | 0 | 93.4% | 20.342 |
| treasures_8_seed_20261023 | alpha_beta/reverse/cache=False | 15487 | 1506 | 2627 | 0 | 85.9% | 42.791 |
| treasures_8_seed_20261023 | alpha_beta/greedy/cache=False | 2190 | 514 | 1028 | 0 | 98.0% | 6.877 |
| treasures_8_seed_20261023 | alpha_beta/greedy/cache=True | 1280 | 345 | 764 | 597 | 98.8% | 4.558 |
| treasures_8_seed_20261024 | minimax/natural/cache=False | 109601 | 0 | 0 | 0 | 0.0% | 283.293 |
| treasures_8_seed_20261024 | minimax/natural/cache=True | 10817 | 0 | 0 | 7224 | 90.1% | 30.047 |
| treasures_8_seed_20261024 | alpha_beta/natural/cache=False | 6548 | 567 | 979 | 0 | 94.0% | 17.958 |
| treasures_8_seed_20261024 | alpha_beta/reverse/cache=False | 5385 | 940 | 1723 | 0 | 95.1% | 15.013 |
| treasures_8_seed_20261024 | alpha_beta/greedy/cache=False | 2343 | 566 | 1127 | 0 | 97.9% | 7.375 |
| treasures_8_seed_20261024 | alpha_beta/greedy/cache=True | 1196 | 349 | 815 | 570 | 98.9% | 4.332 |
| treasures_8_seed_20261025 | minimax/natural/cache=False | 109601 | 0 | 0 | 0 | 0.0% | 281.598 |
| treasures_8_seed_20261025 | minimax/natural/cache=True | 10817 | 0 | 0 | 7224 | 90.1% | 30.144 |
| treasures_8_seed_20261025 | alpha_beta/natural/cache=False | 5551 | 851 | 1532 | 0 | 94.9% | 15.443 |
| treasures_8_seed_20261025 | alpha_beta/reverse/cache=False | 10870 | 1229 | 1758 | 0 | 90.1% | 29.837 |
| treasures_8_seed_20261025 | alpha_beta/greedy/cache=False | 2532 | 605 | 1186 | 0 | 97.7% | 7.958 |
| treasures_8_seed_20261025 | alpha_beta/greedy/cache=True | 1414 | 381 | 860 | 618 | 98.7% | 5.025 |

## 解读边界与报告写作提醒

- 题二原始曼哈顿在 20 个可达格高估，因此不可采纳。但仅E为0、其余格权重≥1的特殊模型仍可证明返回最优，不能据此把违反充分条件等同于必然失败；详见 docs/ALGORITHMS.md。逐点审计见 heuristic_audit.csv。
- heuristic_counterexample.json 给出允许内部零权的扩展反例：raw Manhattan得到3，安全下界和Dijkstra得到2；这是额外场景，不是改写给定题目。
- 优先比较确定性的节点计数。微秒级耗时受操作系统调度、Python解释器和缓存影响；保留IQR，不用单次运行定胜负。
- 外层wall耗时包括入口校验；题三_validate_tree会访问全树检查输入，虽然不计入搜索visited，仍会稀释剪枝带来的端到端加速。algorithm_elapsed_ns仅计博弈递归，可分开分析。
- larger-g仅在f相同时改变顺序，保持最优性；并非任何地图都节省节点。ALT在多查询时可能摊销预处理，须同时检查总成本。
- 题四只有4个宝藏时状态树小，缓存可能无命中或净变慢。6/8宝藏场景用于检验扩展性，不能把最有利种子代替整体表现。
- greedy仅按当前玩家即时收益排序，不改变极大极小目标，也不预先偷看子树的真实值。
- pruned_branches仅是直接未访问兄弟数；visited相对Minimax的减少包含整棵子树效应，启用缓存时还包含缓存节省。
- 拓展是经典方法的正确实现、组合与对照，不声称提出新算法。创新评分需结合理论、证据与适用条件。
