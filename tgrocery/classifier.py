"""模型封装层：把"特征转换器 + 线性分类器"组合成一个可保存/加载的模型。

目录结构（新格式，跨平台、可读、无 pickle）：

.. code-block:: text

    <模型名>/
        id                 # 模型哈希（兼容旧版文件名）
        metadata.json      # 版本、标签、特征配置、分类器参数
        vocab.json         # 特征词表
        vectorizer.npz     # IDF 向量
        model.npz          # 权重矩阵 + 偏置

这种"配置文件 + numpy 数组"的组合有两个好处：

1. 不依赖 Python pickle，模型可以在不同 Python 版本、不同操作系统之间搬运；
2. ``vocab.json`` / ``metadata.json`` 是纯文本，出问题时可以直接打开检查。
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .base import GroceryPredictResult, GroceryTestResult, read_text_src
from .converter import FeatureConfig, GroceryTextConverter
from .learner import LinearClassifier

__all__ = ["GroceryTextModel", "GroceryTest"]

# 模型格式版本。将来若调整目录结构，可据此做迁移。
MODEL_FORMAT_VERSION = 2


class GroceryTextModel:
    """一个完整的文本分类模型。

    Args:
        text_converter: 特征转换器；为 ``None`` 时创建一个默认配置的空转换器。
        model: 训练好的 :class:`LinearClassifier`；加载/保存场景下可为 ``None``。
        classifier_params: 训练参数（``C``、``class_weight`` 等），
            仅用于写入元数据，方便复现实验。
    """

    def __init__(
        self,
        text_converter: GroceryTextConverter | None = None,
        model: LinearClassifier | None = None,
        classifier_params: dict[str, Any] | None = None,
    ) -> None:
        if text_converter is not None and not isinstance(text_converter, GroceryTextConverter):
            raise TypeError("text_converter 必须是 GroceryTextConverter")
        self.text_converter = text_converter or GroceryTextConverter()
        self.model = model
        self.classifier_params = dict(classifier_params or {})
        self._hashcode = str(uuid.uuid4())

    def __str__(self) -> str:
        return f"TextModel instance ({self.text_converter}, {self.model})"

    def __repr__(self) -> str:
        return (
            f"GroceryTextModel(labels={self.get_labels() if self.model else None}, "
            f"n_features={len(self.text_converter.vocab)})"
        )

    def get_labels(self) -> list[str]:
        """返回训练时见过的全部类别名。"""
        if self.model is None:
            return []
        return list(self.model.labels)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, model_name: str | os.PathLike[str], force: bool = False) -> None:
        """保存模型到目录。

        Args:
            model_name: 目标目录。
            force: 目录已存在时是否覆盖。与旧版一致，默认不覆盖并抛错。

        Raises:
            ValueError: 模型尚未训练。
            OSError: 目录已存在且 ``force=False``。
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")

        target = Path(model_name)
        if target.exists():
            if not force:
                raise OSError(f"{target} 已存在；如需覆盖请使用 force=True 或先删除该目录")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        self.text_converter.save(target)
        self.model.save(target)

        metadata = {
            "format_version": MODEL_FORMAT_VERSION,
            "labels": self.model.labels,
            "features": self.text_converter.config.to_dict(),
            "classifier": self.classifier_params,
        }
        with (target / "metadata.json").open("w", encoding="utf-8") as fout:
            json.dump(metadata, fout, ensure_ascii=False, indent=2)

        # id 文件保留旧版命名，便于脚本/工具识别模型目录
        (target / "id").write_text(self._hashcode, encoding="utf-8")

    def load(self, model_name: str | os.PathLike[str]) -> None:
        """从目录加载模型（会覆盖当前实例的内容）。

        Raises:
            ValueError: 目录缺少元数据或格式版本不受支持。
        """
        source = Path(model_name)
        metadata_file = source / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(
                f"{source} 不是有效的 TextGrocery_Shop 模型目录（缺少 metadata.json）。"
                " 旧版（0.1.x）模型无法直接加载，请使用新版重新训练。"
            )

        with metadata_file.open("r", encoding="utf-8") as fin:
            metadata = json.load(fin)

        format_version = int(metadata.get("format_version", 0))
        if format_version > MODEL_FORMAT_VERSION:
            raise ValueError(
                f"模型格式版本 {format_version} 高于当前库支持的 "
                f"{MODEL_FORMAT_VERSION}，请升级 textgrocery-shop"
            )

        labels = [str(label) for label in metadata.get("labels", [])]
        config = FeatureConfig.from_dict(metadata.get("features"))

        converter = GroceryTextConverter(
            custom_tokenize=self.text_converter.custom_tokenize,
            config=config,
        )
        converter.load(source, config=config)

        self.text_converter = converter
        self.model = LinearClassifier.load(source, labels)
        self.classifier_params = dict(metadata.get("classifier", {}))

        id_file = source / "id"
        if id_file.exists():
            self._hashcode = id_file.read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict_text(self, text: str) -> GroceryPredictResult:
        """预测单条文本。

        Args:
            text: 待分类文本。为兼容旧代码，也接受 UTF-8 编码的 ``bytes``。

        Returns:
            :class:`~tgrocery.base.GroceryPredictResult`。

        Raises:
            ValueError: 模型尚未训练。
            TypeError: ``text`` 不是字符串或字节串。
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法预测")

        if isinstance(text, bytes):
            # 旧版允许传 bytes；统一解码成 str 以复用特征处理逻辑
            text = text.decode("utf-8")
        if not isinstance(text, str):
            raise TypeError("The argument should be plain text")

        cols, values = self.text_converter.transform(text)
        label, scores = self.model.predict_one(cols, values)
        return GroceryPredictResult(
            predicted_y=label,
            dec_values=scores,
            labels=self.model.labels,
        )


class GroceryTest:
    """批量测试执行器。

    为了避免"训练集/测试集特征空间不一致"，这里没有重新 fit 转换器，
    而是直接使用模型自带的转换器做 transform —— 未在训练词表中出现的
    特征会被忽略，这也符合线上预测的真实情况。
    """

    def __init__(self, model: GroceryTextModel) -> None:
        self.model = model

    def test(self, text_src: Any, delimiter: str = "\t") -> GroceryTestResult:
        """对列表或文件逐条预测并统计指标。

        Args:
            text_src: ``(标签, 文本)`` 的列表，或 ``标签<delimiter>文本`` 文件。
            delimiter: 文件模式下的分隔符。

        Returns:
            :class:`~tgrocery.base.GroceryTestResult`。
        """
        samples = read_text_src(text_src, delimiter)
        true_y: list[str] = []
        predicted_y: list[str] = []

        for sample in samples:
            try:
                label, text = sample
            except ValueError:
                # 格式不完整的行直接跳过（与原版行为一致）
                continue
            predicted_y.append(str(self.model.predict_text(str(text)).predicted_y))
            true_y.append(str(label))

        return GroceryTestResult(true_y, predicted_y)
