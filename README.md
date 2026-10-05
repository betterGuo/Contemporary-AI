# Contemporary-AI

当代人工智能课程实验仓库。作者：郭宇航（10234507018）。目前收录 Project 1：英文文本分类，包含代码、Conda 环境配置、受控优化、独立复现证据和实验报告。

## Project 1：英文文本分类

对 7,368 条有标签英文文本进行训练，为 2,457 条测试文本预测 `0` 至 `9` 的类别编号。先比较 MultinomialNB、Logistic Regression 和 LinearSVC，再仅使用 LinearSVC，依次开展词级参数搜索、词/字符 TF-IDF 融合消融、五次配对划分验证。

| seed 42 配置 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|
| 原词级基线 | 0.9403 | 0.9403 |
| 优化词级特征 | 0.9457 | 0.9458 |
| 等权词/字符融合 | **0.9484** | **0.9486** |

这些是参与选模的验证指标，不是测试成绩。融合对基线的五次配对 Macro-F1 提升均值为 0.00832，但切分相互重叠，不据此作显著性判断。融合增加计算和存储开销，报告保留了性能与复杂度的权衡。

## 入口与使用

- [安装、运行和复现说明](project1/README.md)
- [两页提交版报告 Markdown](project1/report_submission.md)
- [两页提交版 PDF](project1/output/pdf/Project1_实验报告_郭宇航.pdf)
- [详细实验报告](project1/report.md)
- [最终推荐测试预测](project1/results_optimized/predictions.csv)
- [优化验收记录](project1/optimization_check.json)

```bash
git clone https://github.com/betterGuo/Contemporary-AI.git
cd Contemporary-AI/project1
# 从课程授权来源取得两个数据 CSV，放到当前目录
conda env create -f environment.yml
conda activate project1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python optimize_experiment.py --data-dir . --output-dir results_optimized_new
```

运行需要先安装 Conda。请勿覆盖仓库中已验收的结果目录。原始课程数据和教师要求 PDF 未随仓库公开，也未为课程资料附加再分发许可；数据文件名、格式和复现流程见使用说明。
