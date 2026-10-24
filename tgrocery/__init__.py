"""TextGrocery_Shop 公共入口。

对外 API 与 0.1.x 完全兼容：

.. code-block:: python

    from tgrocery import Grocery

    grocery = Grocery('sample')
    grocery.train([('education', '...'), ('sports', '...')])
    grocery.save()

    new_grocery = Grocery('sample')
    new_grocery.load()
    print(new_grocery.predict('...'))   # 直接打印标签

所有新增能力都通过**仅关键字参数**提供，因此老代码一行都不用改。

Python 版本要求：3.10 及以上（使用 ``X | None`` 联合类型语法与
``dataclasses``）。核心计算只依赖 numpy，安装时无需 C 编译器。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    GroceryException,
    GroceryNotTrainException,
    GroceryPredictResult,
    GroceryTestResult,
    read_text_src,
)
from .classifier import GroceryTest, GroceryTextModel
from .converter import FeatureConfig, GroceryTextConverter
from .learner import LinearClassifier

__version__ = "0.2.0"

__all__ = [
    "Grocery",
    "GroceryException",
    "GroceryNotTrainException",
    "FeatureConfig",
    "__version__",
]


def _materialize_words(
    source: str | Path | Iterable[str] | None,
    label: str,
) -> tuple[str, ...]:
    """把"文件路径或可迭代对象"统一成去空白的词元组。

    Args:
        source: ``None``、词表文件路径（UTF-8，每行一个词），或词的可迭代对象。
        label: 出错信息中显示的字段名（``"stopwords"`` / ``"user_dict"``）。

    Returns:
        去空白后的词元组；真正的去重与大小写处理由
        :class:`~tgrocery.converter.FeatureConfig` 完成。

    Raises:
        GroceryException: 路径不存在或不是文件。
    """
    if source is None:
        return ()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise GroceryException(f"{label} 文件不存在: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = [str(item) for item in source]
    return tuple(line.strip() for line in lines if line.strip())


class Grocery:
    """短文本分类器（训练、保存、加载、预测、测试）。

    使用方式与旧版一致：

    .. code-block:: python

        grocery = Grocery('sample')
        grocery.train(train_src)      # train_src 可以是列表或文件路径
        grocery.save()
        grocery.load()
        grocery.predict('一段文本')
        grocery.test(test_src)

    Args:
        name: 模型名，同时作为 ``save()`` 的目录名。
        custom_tokenize: 自定义分词函数，接收 ``str``，返回 token 可迭代对象。
            为 ``None`` 时使用结巴全模式分词。
        C: 逻辑回归正则化强度的倒数，越大正则越弱。默认 ``8.0``
            （在公开新闻标题数据上明显优于 ``1.0``）。
        class_weight: ``None``、``"balanced"`` 或 ``{标签: 权重}``。
        word_ngrams: 词级 n-gram 阶数，默认 ``(1, 2)``。
        char_ngrams: 字符级 n-gram 阶数，默认 ``(2, 3)``。
            这是改善超短文本/未登录词效果的关键；传 ``()`` 可关闭。
        lowercase: 是否做小写归一化，默认 ``True``。
        min_df: 特征最小文档频率。``None``（默认）表示自动：
            语料少于 1000 篇取 1，否则取 2。
        max_features: 最大特征数，默认不限制。
        binary: 是否使用二值特征（旧版 LibShortText 的默认行为），默认 ``False``。
        sublinear_tf: 是否使用 ``1 + log(tf)``，默认 ``True``。
        use_idf: 是否使用 IDF 加权，默认 ``True``。
        smooth_idf: 是否使用平滑 IDF，默认 ``True``。
        norm: 行归一化方式，默认 ``"l2"``。
        tokenizer: 结巴分词模式，``"full"``（默认，全模式）、
            ``"precise"``（精确模式）或 ``"search"``（搜索引擎模式）。
        normalize: 文本归一化，``"none"``（默认，与旧版一致）或
            ``"nfkc"``（全角转半角，适合全角/中英混排数据）。
        char_ngram_mode: 字符 n-gram 模式，``"global"``（默认，整串滑窗）
            或 ``"word_boundary"``（按词边界滑窗，适合英文/中英混排）。
        mask_patterns: 是否把 URL、邮箱、数字替换为占位符，默认 ``False``。
        stopwords: 停用词文件路径或可迭代对象，只影响词特征。
        user_dict: 用户词典文件路径或可迭代对象，用于补充领域未登录词。
        max_iter: 每个二分类器的最大迭代次数，默认 ``50``。
        tol: 优化器收敛阈值，默认 ``1e-4``。
        verbose: 是否打印训练过程，默认 ``False``。
    """

    def __init__(
        self,
        name: str | Path,
        custom_tokenize: Any = None,
        *,
        C: float = 8.0,
        class_weight: str | dict[str, float] | None = None,
        word_ngrams: Sequence[int] = (1, 2),
        char_ngrams: Sequence[int] = (2, 3),
        lowercase: bool = True,
        min_df: int | None = None,
        max_features: int | None = None,
        binary: bool = False,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        smooth_idf: bool = True,
        norm: str | None = "l2",
        tokenizer: str = "full",
        normalize: str = "none",
        char_ngram_mode: str = "global",
        mask_patterns: bool = False,
        stopwords: str | Path | Iterable[str] | None = None,
        user_dict: str | Path | Iterable[str] | None = None,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        if custom_tokenize is not None and not callable(custom_tokenize):
            raise GroceryException("Tokenize func must be callable.")
        if C <= 0:
            raise GroceryException("C 必须大于 0")
        if max_iter < 1:
            raise GroceryException("max_iter 必须大于等于 1")

        self.name = str(name)
        self.custom_tokenize = custom_tokenize
        self.model: GroceryTextModel | None = None

        self.feature_config = FeatureConfig(
            word_ngrams=tuple(word_ngrams),
            char_ngrams=tuple(char_ngrams),
            lowercase=lowercase,
            min_df=min_df,
            max_features=max_features,
            binary=binary,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            norm=norm,
            tokenizer=tokenizer,
            normalize=normalize,
            char_ngram_mode=char_ngram_mode,
            mask_patterns=mask_patterns,
            stopwords=_materialize_words(stopwords, "stopwords"),
            user_words=_materialize_words(user_dict, "user_dict"),
        )
        self.classifier_params: dict[str, Any] = {
            "C": float(C),
            "class_weight": class_weight,
            "max_iter": int(max_iter),
            "tol": float(tol),
        }
        self.verbose = verbose

        # 兼容旧版属性：旧实现会生成一个临时 .svm 文件并在析构时删除。
        # 新实现直接在内存中组装稀疏矩阵，该属性保留为 None。
        self.train_svm_file: str | None = None

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_load_status(self) -> bool:
        """模型是否已经训练或成功加载。"""
        return (
            self.model is not None
            and isinstance(self.model, GroceryTextModel)
            and self.model.model is not None
        )

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(
        self,
        train_src: str | Path | Iterable[Sequence[Any]],
        delimiter: str = "\t",
    ) -> Grocery:
        """从列表或文件训练模型。

        Args:
            train_src: ``(标签, 文本)`` 的可迭代对象，或
                ``标签<delimiter>文本`` 格式的文件路径（UTF-8）。
            delimiter: 文件模式下的分隔符，默认制表符。

        Returns:
            ``self``，便于链式调用。

        Raises:
            GroceryException: 没有有效训练样本。
        """
        samples = read_text_src(train_src, delimiter)

        texts: list[str] = []
        raw_labels: list[str] = []
        for sample in samples:
            try:
                label, text = sample
            except ValueError:
                # 与原版一致：列数不对的样本直接跳过
                continue
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            if not isinstance(text, str):
                text = str(text)
            texts.append(text)
            raw_labels.append(str(label).strip())

        if not texts:
            raise GroceryException("训练数据为空，无法训练模型")

        # 类别名排序后再编号，保证同样的数据在任何机器上得到一致的
        # 标签下标与模型文件（跨平台可复现）。
        labels = sorted(set(raw_labels))
        label2idx = {label: idx for idx, label in enumerate(labels)}
        y = np.asarray([label2idx[label] for label in raw_labels], dtype=np.int64)

        converter = GroceryTextConverter(
            custom_tokenize=self.custom_tokenize,
            config=self.feature_config,
        )
        converter.fit(texts)
        features = converter.transform_corpus(texts)

        model = LinearClassifier.train(
            features=features,
            y=y,
            labels=labels,
            C=float(self.classifier_params["C"]),
            class_weight=self.classifier_params["class_weight"],
            max_iter=int(self.classifier_params["max_iter"]),
            tol=float(self.classifier_params["tol"]),
            verbose=self.verbose,
        )

        self.model = GroceryTextModel(
            text_converter=converter,
            model=model,
            classifier_params=self.classifier_params,
        )
        return self

    # ------------------------------------------------------------------
    # 预测与测试
    # ------------------------------------------------------------------
    def predict(self, single_text: str) -> GroceryPredictResult:
        """预测单条文本。

        Args:
            single_text: 待分类文本。

        Returns:
            :class:`~tgrocery.base.GroceryPredictResult`。
            其 ``__str__`` 即为标签，因此 ``print(grocery.predict(text))``
            会直接输出类别名（与旧版一致）。
        """
        if not self.get_load_status():
            raise GroceryNotTrainException()
        assert self.model is not None
        return self.model.predict_text(single_text)

    def test(
        self,
        text_src: str | Path | Iterable[Sequence[Any]],
        delimiter: str = "\t",
    ) -> GroceryTestResult:
        """在测试集上评估模型。

        Args:
            text_src: ``(标签, 文本)`` 的列表或文件路径。
            delimiter: 文件模式下的分隔符。

        Returns:
            :class:`~tgrocery.base.GroceryTestResult`。
        """
        if not self.get_load_status():
            raise GroceryNotTrainException()
        assert self.model is not None
        return GroceryTest(self.model).test(text_src, delimiter)

    # ------------------------------------------------------------------
    # 保存与加载
    # ------------------------------------------------------------------
    def save(self) -> None:
        """把模型保存到 ``self.name`` 目录（已存在则覆盖）。"""
        if not self.get_load_status():
            raise GroceryNotTrainException()
        assert self.model is not None
        self.model.save(self.name, force=True)

    def load(self) -> None:
        """从 ``self.name`` 目录加载模型。

        注意：特征配置以模型元数据为准，构造时传入的
        ``word_ngrams`` / ``char_ngrams`` 等参数不会覆盖已保存的配置。
        """
        text_converter = GroceryTextConverter(
            custom_tokenize=self.custom_tokenize,
            config=self.feature_config,
        )
        model = GroceryTextModel(text_converter=text_converter)
        model.load(self.name)
        self.model = model
        # 同步元数据中的配置，便于调用方查询
        self.feature_config = model.text_converter.config
        self.classifier_params = dict(model.classifier_params)

    def __repr__(self) -> str:
        return f"Grocery(name={self.name!r}, trained={self.get_load_status()})"
