"""TextGrocery_Shop 的基础工具与结果对象。

本模块只放"与具体算法无关"的公共设施，目的是让特征工程模块
(:mod:`tgrocery.converter`) 与模型封装模块 (:mod:`tgrocery.classifier`)
都只依赖本模块，从而避免循环导入：

* :func:`read_text_src` —— 统一读取"列表 / 文本文件"两种训练数据源；
* 异常体系 —— :class:`GroceryException` 及其子类；
* :class:`GroceryPredictResult` —— 单条预测结果；
* :class:`GroceryTestResult` —— 批量测试结果（准确率、精确率、召回率）。

设计原则：Python 2 时代的属性名（``accuracy_overall``、``accuracy_labels``、
``recall_labels``、``dec_values``）全部保留，保证老用户的代码无需修改。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "GroceryException",
    "GroceryNotTrainException",
    "GroceryTestResult",
    "GroceryPredictResult",
    "read_text_src",
]


class GroceryException(Exception):
    """TextGrocery_Shop 所有自定义异常的基类。

    继承自 :class:`Exception` 而非更底层的 ``BaseException``，
    这样调用方既可以用 ``except GroceryException`` 精确捕获本库异常，
    也不会意外吞掉 ``KeyboardInterrupt`` 之类的系统信号。
    """


class GroceryNotTrainException(GroceryException):
    """在模型尚未训练/加载时调用 ``predict`` 或 ``test`` 抛出。

    保留 ``message`` 属性是为了兼容 Python 2 时代的使用方式
    （当时 ``Exception`` 并没有像现在这样规范化 ``args``）。
    """

    default_message = "Text model has not been trained."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


def read_text_src(
    text_src: str | Path | Iterable[Sequence[Any]],
    delimiter: str = "\t",
) -> list[Sequence[Any]]:
    """把训练/测试数据源统一转换成"逐行样本"列表。

    支持两种输入：

    1. **文件路径**（``str`` 或 :class:`pathlib.Path`）：
       文件每行格式为 ``标签<delimiter>文本``；空行会被忽略。
       读取时显式使用 UTF-8 编码，避免 Windows 默认 GBK 导致中文乱码。
    2. **可迭代对象**（``list`` / ``tuple`` / 生成器）：
       每个元素应当是 ``(标签, 文本)`` 形式的二元组。

    Args:
        text_src: 数据源，文件路径或 ``(标签, 文本)`` 的可迭代对象。
        delimiter: 文件模式下标签与文本的分隔符，默认为制表符。

    Returns:
        由样本行组成的列表。文件模式下每行被拆成 ``[标签, 文本]`` 两段。

    Raises:
        TypeError: ``text_src`` 既不是路径也不是可迭代对象。
        FileNotFoundError: 文件路径不存在。
        UnicodeDecodeError: 文件不是 UTF-8 编码（本库统一要求 UTF-8）。

    Note:
        这里使用 ``split(delimiter, 1)`` 而不是 ``split(delimiter)``。
        原版用后者，只要正文里再出现一个制表符，整条样本就会被
        ``label, text = line`` 丢弃；限制最多切一刀可以避免这种静默丢数据。
    """
    if isinstance(text_src, (str, Path)):
        path = Path(text_src)
        samples: list[Sequence[Any]] = []
        # newline="" 交给 Python 处理跨平台换行，encoding 固定 UTF-8
        with path.open("r", encoding="utf-8", newline="") as fin:
            for raw_line in fin:
                line = raw_line.rstrip("\r\n")
                if not line:
                    # 空行不参与训练；否则 label/text 拆包会失败
                    continue
                samples.append(line.split(delimiter, 1))
        return samples

    if isinstance(text_src, Iterable):
        # 统一转成 list，便于后续多次遍历（fit / transform 需要多趟）
        return list(text_src)

    raise TypeError("text_src should be a file path (str/pathlib.Path) or an iterable")


class GroceryTestResult:
    """批量测试结果。

    指标定义（与原版保持一致）：

    * ``accuracy_overall``：整体准确率 = 预测正确条数 / 总条数；
    * ``accuracy_labels``：以"预测标签"为分母的精确率（precision），
      即"被判为该类的样本里有多少真的是该类"；
    * ``recall_labels``：以"真实标签"为分母的召回率（recall），
      即"该类真实样本里有多少被找出来"。

    Args:
        true_y: 真实标签列表。
        predicted_y: 预测标签列表，长度必须与 ``true_y`` 一致。

    Attributes:
        accuracy_overall: 整体准确率，取值 ``[0, 1]``。
        accuracy_labels: ``{标签: 精确率}``。
        recall_labels: ``{标签: 召回率}``。
    """

    def __init__(self, true_y: Sequence[str], predicted_y: Sequence[str]) -> None:
        if len(true_y) != len(predicted_y):
            raise ValueError("true_y and predicted_y must have the same length")
        self.true_y = list(true_y)
        self.predicted_y = list(predicted_y)
        self._compute_accuracy_overall()
        self._compute_accuracy_recall_labels()

    def _compute_accuracy_overall(self) -> None:
        """计算整体准确率；测试集为空时定义为 0，避免 ZeroDivisionError。"""
        total = len(self.true_y)
        if total == 0:
            self.accuracy_overall = 0.0
            return
        correct = sum(
            1 for truth, pred in zip(self.true_y, self.predicted_y, strict=True) if truth == pred
        )
        self.accuracy_overall = correct / total

    def _compute_accuracy_recall_labels(self) -> None:
        """逐标签统计 TP / 预测数 / 真实数，再算出精确率与召回率。

        三个计数器放在同一个三元组里：

        * ``counts[pred][0]``：预测正确（TP）
        * ``counts[pred][1]``：被预测为该类的总数（TP + FP）
        * ``counts[truth][2]``：真实属于该类的总数（TP + FN）
        """
        counts: dict[str, list[int]] = {}
        for truth, pred in zip(self.true_y, self.predicted_y, strict=True):
            counts.setdefault(pred, [0, 0, 0])
            counts.setdefault(truth, [0, 0, 0])
            if truth == pred:
                counts[pred][0] += 1
            counts[pred][1] += 1
            counts[truth][2] += 1

        self.accuracy_labels: dict[str, float] = {}
        self.recall_labels: dict[str, float] = {}
        # 排序保证输出顺序稳定，便于测试与日志比对
        for label in sorted(counts):
            tp, predicted_total, true_total = counts[label]
            self.accuracy_labels[label] = tp / predicted_total if predicted_total else 0.0
            self.recall_labels[label] = tp / true_total if true_total else 0.0

    @staticmethod
    def draw_table(
        data: Iterable[Sequence[Any]],
        row_labels: Iterable[str],
        column_labels: Sequence[str],
    ) -> str:
        """把二维数据渲染成固定列宽的空格对齐表格（纯文本，无第三方依赖）。"""
        row_format = "{:<15}" * (len(column_labels) + 1)
        table_string = row_format.format("", *column_labels) + "\n"
        for row_label, row_data in zip(row_labels, data, strict=False):
            table_string += row_format.format(row_label, *row_data) + "\n"
        return table_string

    def show_result(self) -> None:
        """在标准输出打印逐标签的精确率/召回率表格。

        该方法保持原版"打印到终端"的行为；若需要文本结果，
        请直接读取 ``accuracy_labels`` / ``recall_labels`` 或调用
        :meth:`to_table`。
        """
        print(self.to_table())

    def to_table(self) -> str:
        """返回逐标签指标表格字符串（不打印），方便写日志或断言。"""
        labels = [label for label in self.accuracy_labels if label in self.recall_labels]
        return self.draw_table(
            zip(
                [f"{self.accuracy_labels[label] * 100:.2f}%" for label in labels],
                [f"{self.recall_labels[label] * 100:.2f}%" for label in labels],
                strict=True,
            ),
            labels,
            ("accuracy", "recall"),
        )

    def __str__(self) -> str:
        return str(self.accuracy_overall)

    def __repr__(self) -> str:
        return (
            f"GroceryTestResult(accuracy_overall={self.accuracy_overall:.4f}, "
            f"samples={len(self.true_y)})"
        )


class GroceryPredictResult:
    """单条文本的预测结果。

    Attributes:
        predicted_y: 预测出的标签。
        dec_values: ``{标签: 决策值}``。决策值是逻辑回归的原始打分
            （logit，越大越可信），并非概率；概率见 :attr:`probabilities`。
        labels: 训练时出现的全部标签（与 ``dec_values`` 的键一致）。
        probabilities: ``{标签: 概率}``，由 softmax 归一化得到；
            若构造时未提供则尝试从 ``dec_values`` 现算。
    """

    def __init__(
        self,
        predicted_y: str | None = None,
        dec_values: Mapping[str, float] | Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        probabilities: Sequence[float] | None = None,
    ) -> None:
        self.predicted_y = predicted_y
        self.labels = list(labels) if labels is not None else []
        if dec_values is None:
            self.dec_values: dict[str, float] = {}
        elif isinstance(dec_values, Mapping):
            # 兼容旧版：允许直接传入 {标签: 决策值}，此时无需 labels
            self.dec_values = {str(k): float(v) for k, v in dec_values.items()}
            if not self.labels:
                self.labels = list(self.dec_values)
        elif self.labels:
            self.dec_values = dict(zip(self.labels, (float(v) for v in dec_values), strict=False))
        else:
            # 没有标签信息时退化为 "类别序号 -> 分数"，仅为兼容旧行为
            self.dec_values = {str(i): float(v) for i, v in enumerate(dec_values)}

        if probabilities is not None and self.labels:
            self.probabilities = dict(
                zip(self.labels, (float(v) for v in probabilities), strict=False)
            )
        elif self.labels and self.dec_values:
            self.probabilities = _softmax_mapping(self.dec_values)
        else:
            self.probabilities = {}

    def __str__(self) -> str:
        return "" if self.predicted_y is None else str(self.predicted_y)

    def __repr__(self) -> str:
        return f"GroceryPredictResult(predicted_y={self.predicted_y!r})"


def _softmax_mapping(scores: dict[str, float]) -> dict[str, float]:
    """对 ``{标签: logit}`` 做数值稳定的 softmax，返回概率字典。

    算法原理：先减去最大值 ``m``，避免 ``exp`` 上溢；
    ``softmax(z)_i = exp(z_i - m) / sum_j exp(z_j - m)``。
    该结果不改变 argmax，因此预测标签与决策值排序保持一致。
    """
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {label: math.exp(value - max_score) for label, value in scores.items()}
    total = sum(exps.values())
    if total <= 0:  # 理论上不会发生，仅作防御
        return {label: 0.0 for label in scores}
    return {label: value / total for label, value in exps.items()}
