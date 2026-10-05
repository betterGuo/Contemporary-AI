# Project 1 使用说明

本实验使用 TF-IDF 和传统分类器完成十类英文文本分类。目录内包含冻结的实验代码、环境配置、两轮正式运行及独立环境复现结果；不包含模型堆叠或未经运行的模型成绩。

## 1. 文件与结果

| 文件或目录 | 用途 |
|---|---|
| `run_experiment.py` | 第一轮 60 组参数比较、选模、全量训练与预测 |
| `optimize_experiment.py`、`optimization_plan.json` | 冻结的顺序词级搜索、字符融合消融和配对验证 |
| `explore_data.py` | 检查缺失、重复、类别分布和文本长度 |
| `environment.yml` | 跨平台安装入口；具体可用 build 由 Conda 求解 |
| `environment.lock.yml`、`conda-osx-arm64.lock.txt` | 原 macOS ARM64 运行的完整锁定环境 |
| `requirements-lock.txt` | Python 包版本快照，不代替 Conda 原生库 build 锁 |
| `results/`、`results_reproduced/` | 第一轮主运行及独立复现证据 |
| `results_optimized/`、`results_optimized_reproduced/` | 第二轮优化及独立复现证据 |
| `verify_optimization.py`、`optimization_check.json` | 优化的 19 项核验和原运行验收记录 |
| `reproduction_check.py`、`reproduction_check.json` | 第一轮独立复现核验及记录 |
| `report_submission.md`、`output/pdf/` | 学生自述的精简报告及已核验两页 PDF |
| `report.md`、`requirements_check.md` | 详细分析和要求对照 |
| `LOCAL_RUN_NOTES.md` | 原工作区操作记录，路径与部分交付状态描述仅供历史参考 |

推荐提交文件是 **`results_optimized/predictions.csv`**：2,457 行、单列整数标签、无表头和索引，顺序对应测试原行序。SHA-256 为 `c999f912eded693b0a6c78806a7edee1b60680db3f517602f46d5733cf7a42f9`。根目录 `predictions.csv` 是旧基线，故意保留以供核验，不是最终融合预测。

## 2. 准备数据

从课程授权渠道取得原始数据，将下列文件放在本 README 同一目录：

- `train_data.csv`：7,368 行，列名为 `text`、`target`，标签为整数 `0` 至 `9`。
- `test_data_unlabeled.csv`：2,457 行，列名为 `text`。

原始 CSV 和教师要求 PDF 未公开上传。请保持 CSV 原文件不变，不要改行序、另存或清洗后覆盖，否则原数据哈希核验会失败。数据是否可以再次传播应以课程和原来源授权为准。

## 3. 安装独立 Conda 环境

先安装 Miniconda/Conda，然后在能使用 `conda` 的终端运行：

```bash
git clone https://github.com/betterGuo/Contemporary-AI.git
cd Contemporary-AI/project1
conda env create -f environment.yml
conda activate project1
python --version
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

若已创建 `project1`，无需重复创建。原运行使用 Python 3.10.21、NumPy 2.2.6、pandas 2.2.3、SciPy 1.15.2、scikit-learn 1.7.2、Matplotlib 3.10.8 和 threadpoolctl 3.6.0。`environment.yml` 适合其他平台重新求解，但不保证所有底层 build 与原运行一致。

在 macOS Apple Silicon 上，要严格使用原 build，可将上述创建命令替换为：

```bash
conda create --name project1 --file conda-osx-arm64.lock.txt
```

该 explicit lock 不适用于 Windows、Linux 或 Intel Mac。Windows 用户须按所用终端的语法设置线程环境变量，以上 `export` 示例适用于 bash/zsh。

## 4. 数据检查与第一轮基线

```bash
python explore_data.py --output exploration_new.json
python run_experiment.py --data-dir . --output-dir results_baseline_new --submission-path results_baseline_new/predictions.csv
python run_experiment.py --data-dir . --output-dir results_baseline_new --stability-only
python make_report_plots.py --results-dir results_baseline_new
```

第一轮固定 seed 42、分层 80/20 划分，比较词表 5k/10k/20k、unigram/bigram、NB 的三组 alpha、LR 的三组 C 和 LinearSVC 的四组 C。TF-IDF 只在训练折拟合，按验证 Macro-F1 选配置。`--quick` 仅用于流程检查，不代表完整实验。

## 5. 顺序优化与最终预测

```bash
python optimize_experiment.py --data-dir . --output-dir results_optimized_new
python plot_optimization_report.py --results-dir results_optimized_new
```

优化流程按冻结计划依次进行：

1. 只用 LinearSVC，顺序局部搜索词表、`min_df`、`C`，共 8 个唯一候选。
2. 固定赢家，比较字符单支路与四组词/字符融合权重。字符为 `char_wb` 3-5 gram；支路缩放、拼接后归一化，不增加分类器种类。
3. 锁定配置后，在 seed 17、29、53、89、149 上比较四个固定配置，不再次调参，再全量训练和预测。

新预测保存在 `results_optimized_new/predictions.csv`。优化器拒绝非空输出目录；再次运行时请改用另一新目录名，不能覆盖已验收产物。绘图脚本只读取结果，不会训练；报告正文不插入图片。

## 6. 独立环境复现与核验

以下严格复现示例面向 macOS ARM64；再次运行前须确认两个新输出目录均不存在或为空。保留同一终端中的线程变量，创建第二个环境：

```bash
conda create --name project1_reproduce --file conda-osx-arm64.lock.txt
conda activate project1_reproduce
python optimize_experiment.py --data-dir . --output-dir results_optimized_new_reproduced
python verify_optimization.py --primary results_optimized_new --reproduced results_optimized_new_reproduced --output results_optimized_new/optimization_check.json
```

预期输出 `status=passed`、`issues=[]`。核验包含冻结源码/计划、输入与环境锁文件哈希、逐类指标、划分、预测、配对统计及最终文件一致性；时间与机器路径等非确定信息不作字节一致比较。跨平台或不同版本运行可能无法通过严格环境一致性检查，不应伪装为原环境复现。

只检查仓库保存的两轮原运行证据，也需先放入原始数据：

```bash
python verify_optimization.py --output optimization_check_local.json
python reproduction_check.py
```

第一轮核验脚本固定比较 `results/` 和 `results_reproduced/`，不支持自定义目录。保存的 `optimization_check.json` 和 `reproduction_check.json` 是原实验验收记录，不等于在任意新机器上已经复现。保存的日志可能含原机器路径，仅作溯源记录，不是运行前提。

## 7. 结果与局限

原基线验证 Accuracy/Macro-F1 为 0.9403/0.9403；优化词级模型为 0.9457/0.9458；最终等权融合为 0.9484/0.9486。融合比优化词特征提升约 0.00283 Macro-F1，但训练折 CSR 从 10.26 MiB 增至 120.14 MiB；冷路径耗时重建估计从 1.098 秒增至 12.659 秒。CSR 不是进程峰值内存，估计不是独立墙钟计时。

seed 42 参与了选模，五次重复划分共享数据，测试集没有标签。因此报告不声称独立测试准确率、显著性或保证创新加分。精简 PDF 已按 A4、11pt 正文、15mm 页边距导出为两页；原 Markdown 保留以便编辑。
