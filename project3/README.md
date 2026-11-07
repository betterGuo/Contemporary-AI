# 实验三：CIFAR-10 图像分类

**问题：在相同的训练条件下，随机裁剪与翻转、通道注意力各自能改善多少分类准确率？两者叠加是否优于只使用一种？** 四组实验均从头训练，不使用预训练权重。作业要求见[课程讲义](../当代人工智能作业三.pdf)。

## 本次实验的主要发现

三种子平均测试准确率：基线 77.77%，仅增强 79.42%，仅注意力 77.76%，组合 79.46%。在有、无注意力两种条件下分别比较，数据增强在每个种子都提升了准确率，平均主效应 +1.68 个百分点。注意力的平均主效应为 +0.02 个百分点，方向随种子改变。组合的均值比仅增强高 0.04 个百分点，但主实验中组合反而低 0.39 个百分点。

主种子第 8 轮，基线的训练/验证准确率为 92.54%/78.16%，仅增强为 79.95%/79.94%。增强使训练图像更难、同时提高了验证准确率。由于两组训练指标对应不同输入分布，不能用差值直接估算过拟合程度。12 次训练中有 11 次在最后一轮达到验证峰值；这些结果对应固定的 8 轮预算，更长训练是否改变注意力的作用仍值得另做对照。

## 查看结果

- [主实验结果与误差分析](results/analysis/FINDINGS.md)：官方测试集上的四组准确率、定向混淆、逐样本纠正与退化。
- [训练及验证曲线](results/analysis/learning_curves.png) · [混淆矩阵](results/analysis/confusion_matrices.png) · [十类准确率](results/analysis/class_accuracy.csv) · [总体准确率](results/analysis/summary.csv)。
- [跨组同图对照](results/analysis/paired_examples.png) · [逐图预测](results/analysis/paired_predictions.csv) · [每组错例索引](results/analysis/error_samples.csv)；analysis 下的 errors_*.png 为四组各自的典型错例抽样。
- [同图 Grad-CAM](results/explanations/matched_gradcam.png) · [个例解读](解释性分析.md)。
- [三种子配对复跑与效应分析](results/robustness/README.md)，原始训练日志和验证指标见每个实验组的 history.csv。单组 best.pt 为按验证准确率选取的权重，latest.pt 用于同一配置断点续训。

## 方法

| 组 | 训练增强 | 通道注意力 |
|---|---|---|
| baseline | 无 | 无 |
| augment | 随机裁剪、水平翻转 | 无 |
| attention | 无 | 有 |
| both | 随机裁剪、水平翻转 | 有 |

官方训练集按类固定划分：每类 4,500 张训练、500 张验证，总数 45,000/5,000；测试集每类 1,000 张。索引保存在 split.npz。原始 RGB 像素按 x/127.5−1 变换到 [−1,1]，等价于讲义的 ToTensor 后按三个通道均值、标准差 0.5 归一化。训练增强先做 4 像素反射填充，随机裁回 32×32，再以 50% 概率水平翻转。验证和测试只做固定归一化。

CNN 用三阶段卷积（32、64、128 通道）、批归一化、ReLU、池化、全局平均池化和线性分类器。注意力组在第二阶段的 64×8×8 特征上做全局平均，通过 64→8→64 门控层产生通道权重。用 2·sigmoid(z) 并将最后一层初始化为零，初始门值恰为 1；同种子下各组的公共参数、初始输出和每轮样本顺序相同。增加门控后参数从 289,418 增至 290,514。

每组训练 8 轮，batch=256，交叉熵，AdamW（初始学习率 0.002、权重衰减 0.01），每轮余弦衰减。验证准确率最高轮作为最终模型，相同时取最早一轮。同一种子的四组全部选定后才运行测试。主种子 20261107；两次重复只更换训练种子，划分不变，详见 protocol.json。每轮训练曲线是训练模式下随参数更新累计的在线指标，增强组在增强图上计分；验证曲线来自轮末固定权重和未增强图像。

## 复现

本次环境：Python 3.12.6，PyTorch 2.9.1，macOS arm64，Apple GPU（MPS）。依赖见 requirements.txt。下列命令在本目录执行：

```sh
python3 -m pip install -r requirements.txt
python3 prepare_data.py                 # 也可加 --archive /path/to/cifar-10-python.tar.gz
python3 -m unittest discover -s tests -v
python3 train.py                        # 自动选择可用的 MPS、CUDA 或 CPU
for seed in 20261108 20261109; do
  python3 train.py --seed "$seed" --out "results/replicates/$seed" --no-image-copy
done
python3 analyze.py
python3 explain.py
python3 summarize_seeds.py
python3 verify.py
```

prepare_data.py 校验官方压缩包，保留样本和类别顺序，数据摘要在 data/source.json。同配置会从 latest.pt 续训；若要独立重跑，复制项目到新的目录并保留源码和 data，或给 train.py 指定新的 --out。不同后端、数值库版本可能带来浮点差异。

报告可分别引用三类对照：**同种子条件效应**（只改变一个因素）、**固定模型的配对抽样区间**（测试图像波动）、**三个训练种子的均值及样本标准差**（训练波动）。同一批 10,000 张测试图像反复使用，不构成三个独立测试集。
