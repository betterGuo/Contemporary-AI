# Project 1: English text classification

本目录保存文本分类数据、实验脚本和可复现环境文件。任务是将英文文本分为标签 ID `0`--`9`，并按测试集原始行序生成预测 CSV。数据没有提供数字标签到官方类别名称的映射；报告可以根据文本和 Subject 描述可观察到的内容，但不把推测主题当作标签定义。

## 文件

| 文件 | 用途 |
| --- | --- |
| `train_data.csv` | 7,368 条有标签文本，列为 `text` 和 `target` |
| `test_data_unlabeled.csv` | 2,457 条无标签文本，列为 `text` |
| `environment.yml` | 顶层 Conda 依赖版本 |
| `environment.lock.yml` | 锁定版本和 Conda build 的完整环境定义 |
| `conda-osx-arm64.lock.txt` | Apple Silicon/macOS ARM64 的显式 Conda 包清单 |
| `requirements-lock.txt` | 本次环境中 Python distribution 的逐包版本快照（不替代 Conda build lock） |
| `environment_setup.log` | 本次建立独立环境的操作记录 |
| `explore_data.py` | 只读数据摘要与输入有效性检查 |
| `run_experiment.py` | 完整调参、选模、全量重训和预测导出 |
| `make_report_plots.py` | 用保存的结果重画同一组报告图 |
| `reproduction_check.py` | 对比主运行与独立环境复现的关键结果 |
| `reproduction_check.json` | 独立环境一致性核验记录 |
| `optimization_plan.json`、`optimize_experiment.py` | 冻结的词级搜索、字符/融合消融、配对验证和最终预测流程 |
| `verify_optimization.py`、`optimization_check.json` | 两个独立优化环境的结果、逐类指标、划分和提交文件核验 |
| `plot_optimization_report.py` | 只读优化结果并生成报告用的清晰消融图与标注混淆矩阵（不训练） |
| `report.md` | 实验报告 |

## 建立干净环境并运行

本机使用官方 Miniconda 安装；Python 包来自 Conda-forge。下面的 `source` 路径对应本机 Miniconda 安装位置。请先切换到工作区根目录 `/Users/guoyuhang/Desktop/26年秋季学期/当代人工智能`，再执行 `cd " project1/data-Project1"`：目录名开头确实有一个空格。独立环境清单面向 Apple Silicon/macOS ARM64；其他平台不能直接复用该平台 lock。

```bash
source /Users/guoyuhang/.local/miniconda3/etc/profile.d/conda.sh
cd " project1/data-Project1"
conda env create -f environment.yml
conda activate project1
python --version
python -m pip check
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
BASE_OUT="results_baseline_rerun_$(date +%Y%m%d_%H%M%S)"
python run_experiment.py --data-dir . --output-dir "$BASE_OUT" --submission-path "$BASE_OUT/predictions.csv"
python explore_data.py --output "$BASE_OUT/exploration_summary.json"
python run_experiment.py --data-dir . --output-dir "$BASE_OUT" --stability-only
python make_report_plots.py --results-dir "$BASE_OUT"
```

每次正式运行请使用新的输出目录。不要把重跑结果写到已验收的 `results/`、`results_reproduced/`、`results_optimized/` 或 `results_optimized_reproduced/`；这些是留存证据，旧脚本的提交文件也保留为第一轮基线。若需重复一次，使用新的唯一目录名或先将旧目录可恢复地改名归档，不要删除旧证据。

本次独立复现使用另一个 Conda 环境和 explicit lock，结果写入 `results_reproduced/`。在上面设置完线程变量后，运行：

```bash
conda create --name project1_reproduce --file conda-osx-arm64.lock.txt
conda activate project1_reproduce
BASE_REPRO_OUT="results_baseline_reproduced_$(date +%Y%m%d_%H%M%S)"
python run_experiment.py --data-dir . --output-dir "$BASE_REPRO_OUT" --submission-path "$BASE_REPRO_OUT/predictions.csv"
python run_experiment.py --data-dir . --output-dir "$BASE_REPRO_OUT" --stability-only
```

上面的命令把新复现写到唯一目录，以免覆盖归档。冻结的 `reproduction_check.py` 专门比较已验收的标准目录 `results/` 与 `results_reproduced/`，不接受自定义输出目录；它核对环境/输入哈希、60 个候选及逐类 F1、划分和预测明细、预测文件字节及稳定性结果。本项目已保存的基线复现核验通过（`reproduction_check.json` 中 `status=passed`、`issues=[]`）；两次实际运行分别在 `results/` 和 `results_reproduced/`。

第一轮完整默认实验为 60 个候选配置，固定分层 80/20 划分和随机种子 42；模型按验证集 Macro-F1 选择。`--quick` 仅用于检查执行流程，不能替代正式网格结果。稳定性命令读取 `best_config.json` 后仅评估已选定配置，默认使用 seed `13, 42, 73, 101, 137`，不再调参。

独立环境中的主要包版本为 Python 3.10.21、NumPy 2.2.6、pandas 2.2.3、SciPy 1.15.2、scikit-learn 1.7.2、Matplotlib 3.10.8 和 threadpoolctl 3.6.0；完整 Conda build 列在 `environment.lock.yml` 和平台 explicit lock 中，Python 包版本快照列在 `requirements-lock.txt`。该文本清单不包含 Conda 原生库的 build 信息。运行清单会记录输入 CSV 与环境 lock 文件的 SHA-256、脚本 SHA-256、实际命令、完整 Python distributions、解释器/硬件和线程池信息。`run.log` 与 `stability.log` 分开保存。

运行 `python run_experiment.py --help` 可查看数据目录、输出目录和 seed 等选项。两次通过验收的基线运行分别写入 `results/`（环境 `project1`）和 `results_reproduced/`（环境 `project1_reproduce`）。早期系统环境或中断运行的文件单独放在 `results_legacy_system/`、`results_previous_snapshot/`、`results_main_aborted_snapshot/` 和 `results_main_hash_mismatch_snapshot/` 等归档目录；这些目录不是当前结果来源。

## 数据检查与流程

```bash
python explore_data.py
```

主训练会拒绝缺列、空文件、缺失值、空白文本，以及不恰好覆盖整数标签 `0`--`9` 的训练数据。它不删除、不去重或改写 CSV。重复文本与训练/测试重叠数只作为诊断信息记录。

每个候选都先在同一个分层训练子集拟合 TF-IDF，再对验证文本执行 `transform`。向量化参数为 `max_features ∈ {5000,10000,20000}`、`ngram_range ∈ {(1,1),(1,2)}`、`min_df=1`、`sublinear_tf=True`，其余有效默认参数也会写入 `candidate_configurations.json` 和验证结果。固定的 `sublinear_tf=True` 没有被作为实验因素比较，因此本实验不能据此声称它带来提升。

默认比较 MultinomialNB（3 个 `alpha`）、LogisticRegression（3 个 `C`）和 LinearSVC（4 个 `C`），共 60 个设置。图中 `C` 和 `alpha` 曲线按对数横轴呈现；每条线固定其他因素。词表大小图固定分类器强度、`min_df` 与 `sublinear_tf`，只比较 `max_features` 和图例中的 n-gram 设置。所有配置的 Macro-F1 图使用独立数值位置，具体参数请查 `validation_results.csv`。

## 输出与解释边界

完整运行会在结果目录生成逐配置指标、逐类 F1、混淆矩阵、预测明细、全部候选有效参数、数据划分索引、运行清单、警告文本和绘图状态；最终提交文件是一列、无表头 CSV。绘图状态在 `plot_generation.json`，其中只把本次确实保存成功的图列为 `generated`。若某图失败，应以该文件和运行日志为准，不能把旧图当作本次输出。

稳定性分析是在多个分层划分上检查已锁定配置对切分变化的敏感程度。seed 42 的验证结果参与了参数选择，各划分会重叠，因此这些数值不是独立测试分数，也不是无偏泛化评估。测试集没有标签，项目不会声称测得了测试准确率。

第一轮历史基线的 seed 42 最佳配置是 20k unigram TF-IDF 加 `LinearSVC(C=1)`，Accuracy 为 0.9403、Macro-F1 为 0.9403；60/60 个候选成功且没有警告。根目录 `predictions.csv` 及 `results/`、`results_reproduced/` 留作此基线证据，不是后续优化的最终提交。

## 第二轮优化与推荐提交

第二轮冻结第一轮脚本与产物，不新增模型：阶段一最多 8 个唯一 word-unigram 候选；阶段二固定最佳词级配置比较 `char_wb` 3–5 gram、单支路与加权融合；阶段三在五个新分层 seed 上配对比较已锁定的历史基线、word-only、char-only 和融合，不再调参。最终按预定 seed 42 Macro-F1 规则选中 `fusion_w1_c1`（word 80k/min_df 1 + char 80k/min_df 2，`LinearSVC(C=0.5)`），验证 Accuracy 0.9484、Macro-F1 0.9486。独立环境逐项核验通过，见 `optimization_check.json`；原始报告与复现详情见 `report.md`。

本轮建议提交文件是 [`results_optimized/predictions.csv`](results_optimized/predictions.csv)，共 2,457 个标签，SHA-256 `c999f912eded693b0a6c78806a7edee1b60680db3f517602f46d5733cf7a42f9`。它未覆盖根目录的第一轮 `predictions.csv`。融合较最佳 word-only Macro-F1 多约 0.00283，但训练折 CSR 存储约高 11.7 倍、冷路径估计约高 11.5 倍；资源有限时，80k word-only 是更轻量的备选。五 seed 划分重叠且 seed 42 参与了选模，因此配对结果只是描述性稳定性证据，不是独立测试或显著性检验。

若要重新跑优化，请先确保激活 `project1` 并设置上述五个线程变量为 1，再使用唯一的新结果目录；第二个环境也要用不同目录。优化器会拒绝非空输出目录，避免误覆盖：

```bash
OPT_OUT="results_optimized_rerun_$(date +%Y%m%d_%H%M%S)"
python optimize_experiment.py --data-dir . --output-dir "$OPT_OUT"
```

若需独立环境复现，先创建/激活 `project1_reproduce`，再运行同一脚本至另一新目录，然后用两目录执行核验；绘图脚本只读结果，不会重新训练：

```bash
REPRO_OUT="results_optimized_reproduced_$(date +%Y%m%d_%H%M%S)"
python optimize_experiment.py --data-dir . --output-dir "$REPRO_OUT"
python verify_optimization.py --primary "$OPT_OUT" --reproduced "$REPRO_OUT" --output "$OPT_OUT/optimization_check.json"
python plot_optimization_report.py --results-dir "$OPT_OUT"
```

正式排版页数未核验；本项目没有 Word/PDF 交付稿。
