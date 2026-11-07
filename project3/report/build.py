"""Build the five-page report from the measured CIFAR-10 results."""
import csv
from pathlib import Path
from statistics import mean, stdev
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor


REPORT = Path(__file__).resolve().parent
RESULTS = REPORT.parent / "results"
ANALYSIS = RESULTS / "analysis"
GROUPS = ["baseline", "augment", "attention", "both"]
LABELS = ["基线", "仅增强", "仅注意力", "组合"]
SEEDS = [20261107, 20261108, 20261109]
AUTHOR = "郭宇航"
STUDENT_ID = "10234507018"
COURSE_REPOSITORY = "https://github.com/betterGuo/Contemporary-AI"
OUTPUT = REPORT / "实验三 CIFAR-10 图像分类实验报告.docx"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def style_document(doc):
    section = doc.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin = section.right_margin = Mm(22)
    section.top_margin, section.bottom_margin, section.footer_distance = Mm(19), Mm(18), Mm(8)

    for name, size, leading, before, after, bold in [
        ("Title", 23, 34, 0, 9, True),
        ("Heading 1", 17, 26, 0, 9, True),
        ("Heading 2", 13, 21, 7, 5, True),
        ("Normal", 12, 21, 0, 8, False),
        ("Caption", 9.5, 15, 4, 6, False),
        ("Footer", 9, 13, 0, 0, False),
    ]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string("171717")
        style.font.bold, style.font.italic = bold, False
        fonts = style.element.get_or_add_rPr().rFonts
        for attr in list(fonts.attrib):
            if "theme" in attr.lower():
                del fonts.attrib[attr]
        fonts.set(qn("w:eastAsia"), "Heiti SC" if bold else "Songti SC")
        fmt = style.paragraph_format
        fmt.line_spacing, fmt.space_before, fmt.space_after = Pt(leading), Pt(before), Pt(after)
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("当代人工智能实验三　·　")
    page_number = OxmlElement("w:fldSimple")
    page_number.set(qn("w:instr"), "PAGE")
    footer._p.append(page_number)


def new_page(doc, title):
    doc.add_heading(title, 1).paragraph_format.page_break_before = True


def add_author_details(doc):
    details = doc.add_paragraph(style="Caption")
    details.paragraph_format.space_before = Pt(0)
    details.paragraph_format.space_after = Pt(8)
    details.add_run(f"姓名：{AUTHOR}　　学号：{STUDENT_ID}").add_break()
    details.add_run("课程 GitHub 仓库：")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), details.part.relate_to(COURSE_REPOSITORY, RT.HYPERLINK, is_external=True))
    address = details.add_run(COURSE_REPOSITORY)
    address.underline = True
    address.font.color.rgb = RGBColor.from_string("215A7C")
    link.append(address._r)
    details._p.append(link)


def caption(doc, text):
    para = doc.add_paragraph(text, "Caption")
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return para


def figure(doc, path, width_cm, title):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = para.paragraph_format
    fmt.line_spacing, fmt.space_after, fmt.keep_with_next = 1.0, Pt(0), True
    para.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, title)


def table(doc, title, widths_cm, headers, data):
    caption(doc, title).paragraph_format.keep_with_next = True
    tbl = doc.add_table(rows=0, cols=len(headers))
    tbl.autofit, tbl.alignment = False, WD_TABLE_ALIGNMENT.CENTER
    for col, width in zip(tbl.columns, widths_cm):
        col.width = Cm(width)
    borders = OxmlElement("w:tblBorders")
    for side in ["top", "bottom", "left", "right", "insideH", "insideV"]:
        edge = OxmlElement("w:" + side)
        for name, value in [("val", "single"), ("color", "D9D9D9"), ("sz", "4")]:
            edge.set(qn("w:" + name), value)
        borders.append(edge)
    tbl._tbl.tblPr.append(borders)

    for index, values in enumerate([headers, *data]):
        row = tbl.add_row()
        for cell, value, width in zip(row.cells, values, widths_cm):
            cell.width = Cm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.line_spacing = Pt(17)
            para.paragraph_format.space_before = para.paragraph_format.space_after = Pt(2)
            run = para.add_run(str(value))
            run.bold, run.font.size = index == 0, Pt(10)
            fill = OxmlElement("w:shd")
            color = "DFE8ED" if index == 0 else ("F4F6F7" if index % 2 == 0 else "FFFFFF")
            fill.set(qn("w:fill"), color)
            cell._tc.get_or_add_tcPr().append(fill)


def build(output):
    doc = Document()
    style_document(doc)
    p = doc.add_paragraph
    h = lambda title: doc.add_heading(title, 2)

    doc.add_paragraph("实验三　CIFAR-10 图像分类实验报告", "Title")
    add_author_details(doc)
    doc.add_heading("1　实验目的与设计", 1)
    p("在相同的卷积网络和训练预算下，改变输入图像和改变特征通道的权重，哪一种更有帮助？我用三个随机种子各训练四组模型。六次配对中，开启数据增强都提高了测试准确率，平均提高 1.68 个百分点。通道注意力的平均收益只有 0.02 个百分点，换一个种子，变化方向也会不同。")
    h("数据划分与预处理")
    p("CIFAR-10 共有十类 32×32 彩色图像。我从官方训练集每类抽出 500 张作验证，剩余每类 4,500 张用于训练。三个种子共用这份划分，权重选定后在官方测试集上评估（每类 1,000 张）。所有图像按 x/127.5−1 归一化。")
    p("我分别开关数据增强和注意力，得到基线、仅增强、仅注意力和组合四组。增强组先在图像四周反射填充 4 像素，随机裁回 32×32，再以 50% 概率水平翻转。验证和测试均使用完整原图。")
    h("网络与训练设置")
    table(doc, "表 1　卷积网络的三个阶段（尺寸顺序为通道、高、宽）",
          [2.3, 9.2, 4.5], ["阶段", "主要运算", "输出尺寸"],
          [["一", "4×4 卷积（步长 2），3×3 卷积", "32×16×16"],
           ["二", "2×2 最大池化，两次 3×3 卷积，可选通道门控", "64×8×8"],
           ["三", "2×2 最大池化，两次 3×3 卷积", "128×4×4"]])
    p("每次卷积后接批归一化和 ReLU，最后用全局平均池化和线性层输出十类分数。注意力组在第二阶段将每个通道取空间平均，经 64→8→64 的映射和两倍 sigmoid 得到通道乘数。末层从零初始化，初始乘数为 1，增加 1,096 个参数（基线的 0.38%）。")
    p("四组均用 PyTorch 从头训练 8 轮，批大小 256，交叉熵损失，AdamW 初始学习率 0.002、权重衰减 0.01，学习率按轮余弦衰减。同种子下共享层的初始权重和每轮样本顺序相同。按验证准确率选取最佳轮次，并列时取较早一轮。")
    p("实验环境为 Python 3.12.6、PyTorch 2.9.1 和 Apple GPU（MPS）。")

    new_page(doc, "2　四组模型在三个种子下的表现")
    accuracy = {(int(row["seed"]), row["group"]): float(row["test_accuracy_pct"])
                for row in read_rows(RESULTS / "robustness" / "runs.csv")}
    scores = []
    for label, group in zip(LABELS, GROUPS):
        values = [accuracy[seed, group] for seed in SEEDS]
        scores.append([label, *[f"{v:.2f}" for v in values], f"{mean(values):.2f} ± {stdev(values):.2f}"])
    table(doc, "表 2　官方测试集准确率（%，末列为三次训练的均值与样本标准差）",
          [2.6, 3.1, 3.1, 3.1, 4.1], ["方案", *map(str, SEEDS), "均值 ± 标准差"], scores)
    p("我先固定网络，只改变增强开关。普通网络的准确率依次提高 1.55、2.30、1.11 个百分点，带注意力的网络则提高 0.92、2.01、2.17 个百分点。换成只改变注意力开关，组合比仅增强依次低 0.39、高 0.11、高 0.40 个百分点。最高的单次成绩 79.87% 出自组合组，但三次平均只比仅增强高 0.04 个百分点。")
    figure(doc, ANALYSIS / "learning_curves.png", 16.5,
           "图 1　主种子 20261107 的训练与验证曲线（上排为交叉熵损失，下排为准确率）")
    h("训练误差为何给出了不同的排序")
    p("主种子第 1 轮，基线的验证准确率为 57.48%，仅增强为 54.20%。到第 8 轮，基线训练损失更低（0.243 对 0.575），验证损失却更高（0.680 对 0.572），验证准确率也被仅增强反超（78.16% 对 79.94%）。增强组每轮看到的图像更难，但在相同的原图上识别得更准。")
    p("固定主种子的四个模型，对测试图按类做 2,000 次配对重抽样，得到平均效应的 95% 区间。增强为 +0.61 至 +1.83 个百分点，注意力为 −0.49 至 +0.34 个百分点。这个区间反映测试样本的波动，表 2 的标准差反映重新训练带来的波动。")

    new_page(doc, "3　逐类准确率与混淆矩阵")
    class_rows = read_rows(ANALYSIS / "class_accuracy.csv")
    names = [r["class"] for r in class_rows if r["run"] == "baseline"]
    cls_acc = {(r["run"], r["class"]): float(r["accuracy_pct"]) for r in class_rows}
    table(doc, "表 3　主种子四组逐类准确率（%，每类 1,000 张）",
          [2.2] + [1.42] * 10, ["方案", "飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"],
          [[label, *[f"{cls_acc[group, name]:.1f}" for name in names]] for group, label in zip(GROUPS, LABELS)])
    figure(doc, ANALYSIS / "confusion_matrices.png", 15.9,
           "图 2　主种子的测试混淆矩阵，行为真实类别，列为预测类别，格内为图像数")
    p("仅增强比基线多答对 155 张，其中蛙多答对 69 张，鸟却少答对 40 张。猫和狗的误判也有方向之分。与基线相比，组合组把猫认成狗的次数从 158 降到 137，把狗认成猫却从 134 增到 171。猫类准确率上升，狗类仍有更多样本落到了这条分类边界的另一侧。")

    new_page(doc, "4　哪些样本被纠正，哪些又被分错")
    p("我从主种子中抽取了两组样本，一组是组合模型纠正的基线错误，另一组是组合模型新增的错误。把同一张图在四个模型中的预测放在一起，可以看出哪一种设置伴随着预测改变。")
    figure(doc, ANALYSIS / "paired_examples.png", 16.2,
           "图 3　上排为基线错、组合对，下排为基线对、组合错；编号沿用官方测试集顺序")
    p("编号 4468 的图中，红色卡车在画面下半部，上方有大片建筑。基线把它认成汽车，仅注意力组甚至以 0.907 的分数选了船，两组带增强的模型都认出了卡车。随机裁剪会改变车身在画面里的位置，这是模型可能较少依赖固定构图的一个例子。")
    p("编号 565 的白猫图上，结果正好相反。白猫几乎占满画面，耳朵和脸部只剩少量像素。两组未增强模型都认对了，两组带增强的模型却都认成狗。我怀疑部分小图的类别线索靠近边缘，裁剪也可能把区分猫狗所需的细节一起裁掉。")
    h("用水平翻转检验预测的一致性")
    p("我把 5,000 张验证图逐张水平翻转，统计翻转前后预测类别相同的比例。三个种子的平均一致率，基线和仅增强分别为 79.79% 和 86.91%，仅注意力和组合分别为 79.50% 和 86.59%。两种网络都提高约 7 个百分点，说明增强确实减轻了模型对朝向的敏感性。这个指标也会计入两次都判成同一错误类别的图像。")

    new_page(doc, "5　Grad-CAM 与实验小结")
    p("我用四个模型对同一真实类别的分数求梯度，在第二阶段的 8×8 特征图上计算 Grad-CAM。每行对应一张测试图，各热图独立归一化，适合比较响应位置。")
    figure(doc, RESULTS / "explanations" / "matched_gradcam.png", 12.7,
           "图 4　左起为原图、基线、仅增强、仅注意力、组合；后四列标明预测类别")
    p("鸟图 #25 中，基线把鸟认成狗。组合组答对了，亮区却落在画面右缘。狗图 #24 中，几组在头部或前身都有响应，四组仍全部预测为鹿。即使找到了主体，模型也可能只认出了几类动物共有的轮廓。")
    p("门控给整个通道乘以同一权重。主种子测试图上，门值与初始值 1 的平均绝对偏差在仅注意力组为 0.0431，在组合组为 0.0455。通道权重确实变了，放大的也可能是几个类别共有的特征。")
    h("实验体会")
    p("如果只看最高的 79.87%，我会选组合模型。可在主种子里，给增强模型加上注意力，纠正了 377 张，又把原本答对的 416 张分错了。以后比较模型，我会把纠正和新增的错例都抽出来看。这次的 8 轮预算下，我更愿意用六次配对都带来提升的数据增强。")
    p("12 次训练有 11 次在最后一轮才取得最高验证准确率。我想用相同的三个种子延长四组训练，再看注意力的收益是否随轮数改变。")
    doc.core_properties.title = "实验三 CIFAR-10 图像分类实验报告"
    doc.core_properties.author = AUTHOR
    doc.save(output)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
