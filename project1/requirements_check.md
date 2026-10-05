# 实验要求与证据核对

本表核对两轮正式 Conda 运行及各自的独立环境复现。第一轮基线留在 `results/` 与 `results_reproduced/`，第二轮受控优化留在 `results_optimized/` 与 `results_optimized_reproduced/`。历史、中断或 hash 不匹配的运行另行归档。验证集用于选模，以下验证指标都不是独立测试结果；推荐提交文件是 `results_optimized/predictions.csv`，根目录 `predictions.csv` 仍是第一轮基线。

| 要求/评分点 | 实现与本次证据 | 状态 |
|---|---|---|
| 使用至少两种分类算法 | MultinomialNB、LogisticRegression、LinearSVC 三种模型均参与完整网格比较。 | 已完成 |
| 比较表示和模型参数 | 词表上限 5k/10k/20k × unigram/unigram+bigram 共 6 种特征配置；NB 有 3 个 alpha，LR 有 3 个 C，LinearSVC 有 4 个 C，共 60 个候选。 | 已完成，60/60 成功 |
| 固定随机性并分层划分 | 主实验采用 `test_size=0.20`、`random_state=42`、分层划分；数据量为 5,894 条训练折和 1,474 条验证折。 | 已完成 |
| 验证集用于模型选择 | 以 Macro-F1 为主指标选择配置；完全平分时保留固定网格顺序中先出现的候选。seed 42 的验证划分参与了选模。 | 已完成；不得称为独立测试 |
| 避免预处理泄漏 | 先划分数据，每个候选只在训练折 `fit_transform` TF-IDF，验证折只 `transform`；选定配置后才在全部 7,368 条有标签数据上重训。 | 已完成 |
| 输入数据有效性 | 训练/测试 CSV 所需列和数据行完整；训练标签为整数 ID `0`--`9` 且十类齐全；文本无缺失或空白。EDA 还记录精确重复和跨文件精确重叠均为 0。 | 已完成，摘要见 `results/exploration_summary.json` |
| 保留原始数据 | 没有删除、去重、截断或改写 CSV；字符长度中位数为 1,139，最长 160,616，长文本保留。 | 已完成 |
| 生成测试集预测 | 第一轮提交在根目录 `predictions.csv`；第二轮优化提交在 `results_optimized/predictions.csv`。两者均为单列、无表头/索引、2,457 行，标签为 `0`--`9`，按测试 CSV 原行序排列；第二轮没有覆盖基线文件。 | 已完成；建议提交第二轮文件 |
| 完整、可追踪的运行信息 | 第一轮清单记录 60 个候选；第二轮清单记录受控搜索/配对配置、有效参数、CSV/锁文件与代码 SHA-256、实际命令、解释器/硬件/线程池、耗时及 warnings。 | 已完成，见两轮 `run_config.json`、日志和 `optimization_results.json` |
| 可重复划分敏感性 | 将主运行已选配置固定，在分层 seed `13, 42, 73, 101, 137` 上评估；不在这些划分上重新调参。拆分索引和锁定配置写入稳定性产物。 | 已完成；仅为划分敏感性分析 |
| 图表 | 全配置分数、单因素 C、单因素 alpha、固定其余条件的词表大小对照、最佳混淆矩阵等 7 张图均由本次脚本生成；状态文件没有失败项。 | 已完成，见 `results/plot_generation.json` |
| 误分类分析 | 第二轮验证例：原始行 3520，Subject `Conner CP3204F info please`，真实 2，历史基线预测 7、优化预测 2；原始行 7351，Subject `Re: images of earth`，真实 0，历史基线预测 0、优化预测 5。数字 ID 不映射为未经证实的类别名称。 | 已完成，详见 `report.md` |
| 独立环境复现 | 第一轮第二个 Conda 环境重跑 60 个配置和五 seed 稳定性分析；第二轮独立环境重跑新搜索和五 seed 配对分析。各自的预测字节和验证/逐类指标与相应主运行一致，运行耗时允许不同。 | 已完成；`reproduction_check.json`、`optimization_check.json` 均为 `passed` 且 issues 为空 |
| 报告页数/正式排版 | 本轮没有生成 Word/PDF 正式排版稿，也没有进行页数验收。 | 未核验 |

## 实测结果摘要

以下先记录第一轮基线，第二轮优化结果随后单独列出；二者不要混称。

- 主实验：60/60 候选成功，无 warning。最佳配置为 20k unigram TF-IDF + `LinearSVC(C=1)`，验证 Accuracy `0.9402985`，Macro-F1 `0.9403411`。
- 各模型按 Macro-F1 选出的最佳配置：MultinomialNB `alpha=0.1`（Accuracy `0.9172320`，Macro-F1 `0.9179939`）；LogisticRegression `C=10`（`0.9375848`，`0.9377009`）；LinearSVC `C=1`（`0.9402985`，`0.9403411`）。三者均为 20k unigram 特征。
- 锁定最佳配置的五 seed Macro-F1 均值 `0.9421720`、样本标准差 `0.0049291`、范围 `0.9343907`--`0.9466281`；Accuracy 均值 `0.9420624`、标准差 `0.0049530`、范围 `0.9341927`--`0.9464043`。验证拆分相互重叠，且 seed 42 已用于选模；这些结果不构成独立或无偏测试估计。
- 全量重训阶段：向量化 `0.587816` 秒，测试集转换 `0.177659` 秒，估计器拟合 `0.251247` 秒，预测 `0.001422` 秒，合计 `1.018174` 秒，无 warning。完整主流程墙钟时间约 `105.268` 秒，该数值包含 60 个候选的评估和图表生成，并非网格训练时间。
- 最终主运行和独立环境预测内容相同；预测提交文件均为 2,457 行。环境间耗时存在微小差异是正常的。

## 第二轮受控优化核对

| 要求/检查点 | 本次实现与证据 | 状态 |
|---|---|---|
| 不改动第一轮证据 | `run_experiment.py`、`reproduction_check.py`、第一轮结果目录及根 `predictions.csv` 保持原样。新代码、计划和输出独立保存。 | 已完成；新核验确认旧 reproduction 仍通过、旧提交 SHA 仍为 `308c2a…da1c` |
| 预先限定的搜索 | 阶段一 LinearSVC unigram 单因素顺序搜索：`max_features={20k,40k,80k,None}`，再 `min_df={1,2,3}`，再 `C={1,0.5,2}`，最多 8 个唯一拟合，固定种子 42；不枚举全组合，不声称全局最优。 | 已完成，见 `optimization_plan.json` 和 `selection_manifest.json` |
| 词/字符消融 | 阶段二固定最佳词级 `80k/min_df=1/C=0.5`，比较 word-only、`char_wb` 3–5 gram 80k/min_df2、及四个加权融合；各支路 L2，融合后拼接并 L2 归一化。seed42 按 Macro-F1 锁定 fusion `1:1`。 | 已完成；stage2 六个表示/权重候选及历史 baseline 均记录 |
| 配对敏感性 | 阶段三只比较预先锁定的 baseline、best word、char-only、best fusion；seed `17,29,53,89,149`，各 seed 先逐样本 split 评分并相对 baseline 求 delta，再对五个 delta 求均值/样本 SD。没有在该阶段调参。 | 已完成；描述性结果，不作 CI/显著性结论；不同划分互有重叠 |
| 选择与测试封存 | 最终 `fusion_w1_c1` 在阶段三开始前锁定；只有全量最终拟合预测时才读取未标注测试文本，测试集无标签、未用于选择。 | 已完成；见锁定 selection manifest 及 `optimization_check.json` |
| 全量优化预测 | 推荐提交 `results_optimized/predictions.csv`，2,457 行，SHA-256 `c999f912eded693b0a6c78806a7edee1b60680db3f517602f46d5733cf7a42f9`。独立环境输出完全一致，未覆盖根目录 baseline 文件。 | 已完成 |
| 独立优化复现核验 | 核对两环境 37 组训练/验证指标、所有逐类分数、预测行、划分索引、配对 delta、CSR 存储与系数形状、锁/代码/输入哈希和提交文件；19 项检查均通过。 | 已完成，见 `optimization_check.json`，`issues=[]` |
| 图表和错例 | 保存 seed42 表示消融、阶段三配对差值、标注混淆矩阵以及两环境原始图状态。错误数 88→76，其中纠正 24 条、引入 12 条；不宣称消除类别 7→2 混淆。 | 已完成，见 `results_optimized/` 与 `report.md` |
| 资源代价 | 融合 Macro-F1 较最佳 word-only 高约 0.002835；训练折 CSR 约 120.14 MiB 对 10.26 MiB，冷路径总耗时重建估计 12.659 s 对 1.098 s。估算非独立 wall time，CSR 非 RSS。 | 已核对；报告权衡，不称无成本提升 |

- 第二轮 seed42：历史 baseline Accuracy/Macro-F1 `0.940299/0.940341`；word-only `0.945726/0.945784`；最终 fusion `0.948440/0.948619`。字符单支路为 `0.935550/0.935915`。
- 五个配对 seed 的 Macro-F1：baseline `0.939640 ± 0.005388`；best word `0.943992 ± 0.006648`，配对 Δ `+0.004353 ± 0.002686`；char-only `0.939193 ± 0.008105`，配对 Δ `−0.000447 ± 0.006392`；best fusion `0.947962 ± 0.004971`，配对 Δ `+0.008322 ± 0.003328`。这是切分敏感性描述；重复拆分相互重叠且 seed42 已用于选模，不是独立或无偏测试。
- 具体验证错例见 `report.md`：原始行 3520（真实 2，7→2 纠正）、2607（真实 4，3→4 纠正）、7351（真实 0，0→5 新错）。ID 仍不映射为未经证实的类别名。
- 数据长度分组使用全部有标签文本计算的四分位切点 `737.75, 1139, 1773.5`；Q1 的 366 条验证样本准确率由 `0.8743` 到 `0.8962`。这是描述性分组结果，不作因果解释。

## 环境和复现证据

- 独立环境名为 `project1`，Python `3.10.21`；NumPy `2.2.6`、pandas `2.2.3`、SciPy `1.15.2`、scikit-learn `1.7.2`、Matplotlib `3.10.8`、threadpoolctl `3.6.0`。运行平台为 macOS ARM64，记录的数值库线程限制为 1。
- 本机通过官方 Miniconda 安装器安装 Conda；本实验 Python 包来自 Conda-forge。`environment.yml` 说明顶层依赖，`environment.lock.yml` 锁定包版本/build，`conda-osx-arm64.lock.txt` 为该平台 explicit lock；`requirements-lock.txt` 是 Python distribution 版本快照，不含 Conda 原生包的 build 信息。
- 命令中设置了 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS`、`NUMEXPR_NUM_THREADS` 为 1。实验命令、包清单、输入和环境锁 SHA-256、脚本 SHA-256、硬件与线程池信息均记录在运行清单及日志中。
- 第二个环境 `project1_reproduce` 从 `conda-osx-arm64.lock.txt` 创建，输出在 `results_reproduced/`。可运行 `python reproduction_check.py --output reproduction_check.json` 核对主运行与复现；该脚本检查环境/输入哈希、候选与逐类指标、切分/预测明细、预测文件字节及五 seed 结果，忽略合理不同的耗时字段。
- `results_legacy_system/`、`results_previous_snapshot/`、`results_main_aborted_snapshot/` 和 `results_main_hash_mismatch_snapshot/` 均为历史或未验收归档，绝不能与当前 `results/`、`results_reproduced/` 的结论混用。

## 解释边界

- `sublinear_tf=True` 对所有候选固定，没有被单独比较；不能声称已实证其提升或把它说成验证过的创新。
- unigram 与 unigram+bigram 的分数对照是实测；“固定词表大小下 bigram 占用名额、挤出 unigram”只是待检验解释，并未由本实验验证因果。
- 第二轮只做预设路径的局部搜索；stage 1 不含 bigram，也不能说明没有搜索的参数组合会更差。字符支路的收益须结合更高的特征存储和拟合成本解释。
- 实际运行的是 LinearSVC，不是 `SVC(kernel="linear")`；LR 使用 L2 与 `lbfgs`，LinearSVC 使用 squared-hinge/L2 并按 one-vs-rest 处理多类，`SVC(kernel="linear")` 基于 LIBSVM 且采用 one-vs-one。MLP 仅作理论讨论，无训练结果。
- 对句向量做 Word2Vec 平均不能保留词序；BERT 一类上下文模型能利用位置和上下文，但没有在本次网格中运行。
- 课程创新/优化得分由教师依据课程标准判断；本核对表不保证创新加分。
