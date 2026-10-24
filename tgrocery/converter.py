"""文本特征工程：分词、n-gram 展开、TF-IDF 加权与稀疏向量化。

为什么需要单独一层"特征工程"？

TextGrocery_Shop 的使用场景是**短文本分类**（新闻标题、商品标题、工单摘要）。
这类文本有两个特点：

1. 长度很短，往往只有十几个字，单个词的区分度有限；
2. 词汇极度稀疏，未登录词、错别字、中英混排非常常见。

原版只使用"结巴分词 + 词 unigram/bigram 计数 + 二值化 + L2 归一化"。
在标题级文本上尚可，但文本再短一些时，词粒度特征会大量落在词表之外，
线性模型的输入直接变成近乎全零的向量。

本模块在保留原有点思路的基础上增加了几项对短文本非常有效的处理：

* **字符 n-gram（默认 2~3）**：不依赖分词结果，直接对字符滑窗。
  中文里"高考/考生/考试"这类共享字符的词语可以通过 ``2-gram``
  建立联系，即使整词未登录也能获得有效特征；中英混排同样适用。
* **sublinear TF（``1 + log(tf)``）**：抑制高频词在同一文档内的重复计数。
  原版直接用次数，长一点的文本会被某个高频词主导。
* **平滑 IDF**：``log((1 + N) / (1 + df)) + 1``。
  一方面降低"到处都出现"的虚词权重，另一方面避免除零和负值。
* **L2 行归一化**：让不同长度文本的向量处于可比的尺度上。
* **min_df / max_features**：过滤只出现一两次的噪声特征，控制内存与过拟合。

最终产出是 CSR 风格的三个 numpy 数组 ``(rows, cols, data)``，
供 :mod:`tgrocery.learner.linear` 做稀疏矩阵乘法，全程不需要编译 C 扩展。
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
import numpy as np

__all__ = ["FeatureConfig", "SparseMatrix", "GroceryTextConverter"]

# 词特征与字符特征的前缀。加前缀是为了避免 "w:a" 与 "c:a" 这类
# 来自不同特征空间的键发生碰撞。
_WORD_PREFIX = "w:"
_CHAR_PREFIX = "c:"

# 词 n-gram 内部的连接符，仅用于生成可读、可持久化的特征名。
_NGRAM_JOINER = "\x1f"

# 可选的占位符替换规则。只在 ``mask_patterns=True`` 时启用，
# 目的是把"每条样本都不同"的 URL / 邮箱 / 数字折叠成稳定特征。
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER_RE = re.compile(r"\d+(?:[.,:/\-]\d+)*")


@dataclass(frozen=True)
class FeatureConfig:
    """特征工程配置。

    Attributes:
        word_ngrams: 词级 n-gram 阶数集合。默认 ``(1, 2)``，
            即同时使用 unigram 和 bigram（与原版行为一致）。
            传 ``()`` 可完全关闭词特征。
        char_ngrams: 字符级 n-gram 阶数集合。默认 ``(2, 3)``。
            字符 n-gram 对超短文本、未登录词和中英混排最有效。
            传 ``()`` 可关闭（此时行为接近原版）。
        lowercase: 是否对文本做小写归一化。中文不受影响，
            主要是让英文 ``Apple`` 与 ``apple`` 共享特征。
        min_df: 特征最小文档频率。小于该值的特征会被丢弃。
            ``None`` 表示"自动"：语料少于 1000 篇时取 1（保留全部特征，
            适合小样本演示），否则取 2（过滤只出现一次的噪声特征，
            在大语料上通常能同时提升准确率与速度）。
        max_features: 按文档频率保留的最大特征数（``None`` 表示不限制）。
        binary: 是否把所有非零特征统一置为 1（二值特征）。
            这是旧版 LibShortText 的默认行为（``-D 1``）。
        sublinear_tf: 是否使用 ``1 + log(tf)`` 代替原始词频。
            仅当 ``binary=False`` 时生效。
        use_idf: 是否乘以 IDF 权重。
        smooth_idf: 是否使用平滑 IDF 公式。
        norm: 行归一化方式，``"l2"`` 或 ``None``。
        tokenizer: 结巴分词模式，``"full"``（全模式，默认，与原版一致）、
            ``"precise"``（精确模式）或 ``"search"``（搜索引擎模式）。
            实测在新闻标题上全模式效果最好；精确模式特征更少、训练略快。
        char_ngram_mode: 字符 n-gram 的切分方式。``"global"``（默认）
            在去掉空白后的整串上滑窗；``"word_boundary"`` 只在分词结果
            内部滑窗并补边界空格（类似 scikit-learn 的 ``char_wb``），
            可避免英文/中英混排产生跨词噪声，但分词更重时特征也更多。
        normalize: 文本归一化方式。``"none"``（默认，保持与旧版一致）；
            ``"nfkc"`` 把全角字符转成半角、统一兼容字符，适合含有
            全角字母/数字或中英混排的数据。中文汉字不受影响。
        mask_patterns: 是否把 URL、邮箱、数字替换为 ``__url__`` /
            ``__email__`` / ``__num__`` 占位符。默认关闭；对含大量
            链接、价格、日期的 UGC 文本可减少特征爆炸。
        stopwords: 停用词集合。只作用于词特征和 ``word_boundary``
            字符特征；``global`` 字符特征不删除停用词，否则会破坏
            相邻字符 n-gram。
        user_words: 用户词典中的词。初始化时会加入结巴词典，
            用于补充领域实体、产品名等未登录词；会随模型一起保存。
    """

    word_ngrams: tuple[int, ...] = (1, 2)
    char_ngrams: tuple[int, ...] = (2, 3)
    lowercase: bool = True
    min_df: int | None = None
    max_features: int | None = None
    binary: bool = False
    sublinear_tf: bool = True
    use_idf: bool = True
    smooth_idf: bool = True
    norm: str | None = "l2"
    tokenizer: str = "full"
    char_ngram_mode: str = "global"
    normalize: str = "none"
    mask_patterns: bool = False
    stopwords: tuple[str, ...] = ()
    user_words: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "word_ngrams", tuple(int(n) for n in self.word_ngrams))
        object.__setattr__(self, "char_ngrams", tuple(int(n) for n in self.char_ngrams))

        for n in self.word_ngrams + self.char_ngrams:
            if n < 1:
                raise ValueError("n-gram 阶数必须是大于等于 1 的整数")
        if self.min_df is not None and self.min_df < 1:
            raise ValueError("min_df 必须大于等于 1（或 None 表示自动）")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features 必须是正整数或 None")
        if self.norm not in (None, "l2"):
            raise ValueError("norm 只支持 'l2' 或 None")
        if self.tokenizer not in ("full", "precise", "search"):
            raise ValueError("tokenizer 只支持 'full'、'precise' 或 'search'")
        if self.char_ngram_mode not in ("global", "word_boundary"):
            raise ValueError("char_ngram_mode 只支持 'global' 或 'word_boundary'")
        if self.normalize not in ("none", "nfkc"):
            raise ValueError("normalize 只支持 'nfkc' 或 'none'")

        # 停用词/用户词统一去空白、去重；开启 lowercase 时一并转小写，
        # 否则大写词典词在转小写后的文本里永远匹配不上。
        def prepare(words: Sequence[str]) -> tuple[str, ...]:
            cleaned: list[str] = []
            for word in words:
                text = str(word).strip()
                if not text:
                    continue
                cleaned.append(text.lower() if self.lowercase else text)
            return tuple(dict.fromkeys(cleaned))

        object.__setattr__(self, "stopwords", prepare(self.stopwords))
        object.__setattr__(self, "user_words", prepare(self.user_words))

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典，用于写入模型元数据。"""
        return {
            "word_ngrams": list(self.word_ngrams),
            "char_ngrams": list(self.char_ngrams),
            "lowercase": self.lowercase,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "binary": self.binary,
            "sublinear_tf": self.sublinear_tf,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "norm": self.norm,
            "tokenizer": self.tokenizer,
            "char_ngram_mode": self.char_ngram_mode,
            "normalize": self.normalize,
            "mask_patterns": self.mask_patterns,
            "stopwords": list(self.stopwords),
            "user_words": list(self.user_words),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FeatureConfig:
        """从模型元数据恢复配置；缺省字段使用默认值，便于向前兼容。"""
        if not data:
            return cls()
        return cls(
            word_ngrams=tuple(data.get("word_ngrams", (1, 2))),
            char_ngrams=tuple(data.get("char_ngrams", (2, 3))),
            lowercase=bool(data.get("lowercase", True)),
            min_df=int(data["min_df"]) if data.get("min_df") is not None else None,
            max_features=data.get("max_features"),
            binary=bool(data.get("binary", False)),
            sublinear_tf=bool(data.get("sublinear_tf", True)),
            use_idf=bool(data.get("use_idf", True)),
            smooth_idf=bool(data.get("smooth_idf", True)),
            norm=data.get("norm", "l2"),
            # 旧模型没有这些字段：使用"旧行为"作为默认值，
            # 保证 0.2.0 早期模型仍能按原特征流程推理。
            tokenizer=str(data.get("tokenizer", "full")),
            char_ngram_mode=str(data.get("char_ngram_mode", "global")),
            normalize=str(data.get("normalize", "none")),
            mask_patterns=bool(data.get("mask_patterns", False)),
            stopwords=tuple(data.get("stopwords") or ()),
            user_words=tuple(data.get("user_words") or ()),
        )


@dataclass
class SparseMatrix:
    """CSR（按行压缩）风格的稀疏矩阵。

    这里不使用 ``scipy.sparse``，原因是：

    * scipy 体积大、安装重，而本库希望 ``pip install`` 后即可用；
    * 我们只需要"矩阵乘向量"和"转置乘向量"两种运算，
      用 :func:`numpy.bincount` 就能以 C 语言速度完成，
      没有必要为一个可选优化引入重量级依赖。

    Attributes:
        rows: 每个非零元素所在的行号（``int32``）。
        cols: 每个非零元素所在的列号（``int32``）。
        data: 每个非零元素的取值（``float32``）。
        shape: ``(行数, 列数)``。
    """

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.rows.shape != self.cols.shape or self.rows.shape != self.data.shape:
            raise ValueError("rows / cols / data 的长度必须一致")
        if self.rows.size and int(self.cols.max()) >= self.shape[1]:
            raise ValueError("列号超出特征维度")

    @property
    def nnz(self) -> int:
        """非零元素个数。"""
        return int(self.data.size)

    def __len__(self) -> int:
        return int(self.shape[0])


class GroceryTextConverter:
    """文本 -> 稀疏特征向量 的转换器（fit / transform 两阶段）。

    工作流程（与 scikit-learn 的 ``TfidfVectorizer`` 思路一致）：

    1. :meth:`fit`：扫描全部训练文本，统计每个特征的文档频率（DF），
       过滤 ``min_df`` / ``max_features``，建立 ``特征名 -> 列号`` 的词表，
       并计算 IDF；
    2. :meth:`transform`：把单条文本转换为 ``(列号, 取值)``；
    3. :meth:`transform_corpus`：批量转换，拼成 :class:`SparseMatrix`。

    为什么必须先 fit 再 transform：列号必须与训练时的权重矩阵严格对齐，
    否则模型学到的权重会应用到错误的特征上。

    Args:
        custom_tokenize: 自定义分词函数，接收 ``str`` 返回可迭代的 token。
            为 ``None`` 时使用结巴的"全模式"分词（``cut_all=True``），
            与旧版默认行为一致。
        config: 特征配置；为 ``None`` 时使用 :class:`FeatureConfig` 默认值。
    """

    def __init__(
        self,
        custom_tokenize: Any = None,
        config: FeatureConfig | None = None,
    ) -> None:
        if custom_tokenize is not None and not callable(custom_tokenize):
            raise TypeError("custom_tokenize 必须是可调用对象")
        self.custom_tokenize = custom_tokenize
        self.config = config or FeatureConfig()
        # 词表：列表下标即特征列号；字典用于 O(1) 查找
        self.vocab: list[str] = []
        self.feature2idx: dict[str, int] = {}
        # IDF 向量，长度与词表一致；未启用 IDF 时为 None
        self.idf: np.ndarray | None = None
        self._fitted = False
        self._stopwords = frozenset(self.config.stopwords)

        # 把用户词典注册到结巴。jieba 的词典是进程级全局状态，
        # add_word 可重复调用且开销很小，这里按需注册一次。
        self._register_user_words()

    def _register_user_words(self) -> None:
        """把 ``config.user_words`` 注册进结巴词典（幂等）。"""
        for word in self.config.user_words:
            jieba.add_word(word)

    # ------------------------------------------------------------------
    # 0. 文本归一化
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        """做与分词无关的文本归一化。

        * NFKC 把全角字母/数字/标点转成半角，统一 ``ＡＢＣ`` 与 ``ABC``；
        * 可选的占位符替换把 URL、邮箱、数字折叠成稳定特征，
          避免价格、日期、编号制造大量只出现一次的特征；
        * 最后统一小写并去掉首尾空白。
        """
        normalized = text
        if self.config.normalize == "nfkc":
            normalized = unicodedata.normalize("NFKC", normalized)
        if self.config.mask_patterns:
            normalized = _URL_RE.sub(" __url__ ", normalized)
            normalized = _EMAIL_RE.sub(" __email__ ", normalized)
            normalized = _NUMBER_RE.sub(" __num__ ", normalized)
        normalized = normalized.strip()
        if self.config.lowercase:
            normalized = normalized.lower()
        return normalized

    # ------------------------------------------------------------------
    # 1. 原始特征抽取
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        """按配置切词，并过滤空白与停用词。

        三种结巴模式：

        * ``full``：全模式，输出所有可能成词的片段（原版默认）。
          召回高，但会有重叠词；适合短文本。
        * ``precise``：精确模式，只输出一条最可能的切分。
        * ``search``：搜索引擎模式，在精确模式基础上补充长词子词。

        注意：``cut_all=True`` 会输出空格 token，旧版直接把它们
        计入字典，既浪费特征维度又没有任何区分度，这里统一过滤。
        """
        if self.custom_tokenize is not None:
            tokens = self.custom_tokenize(text)
        elif self.config.tokenizer == "full":
            tokens = jieba.cut(text, cut_all=True)
        elif self.config.tokenizer == "precise":
            tokens = jieba.cut(text, cut_all=False)
        else:
            tokens = jieba.cut_for_search(text)

        cleaned: list[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(f"分词器返回了非字符串 token: {type(token)!r}")
            token = token.strip()
            if token and token not in self._stopwords:
                cleaned.append(token)
        return cleaned

    def _extract_counts(self, text: str) -> dict[str, int]:
        """抽取一条文本的"原始特征 -> 出现次数"。

        特征分为两个空间：

        * ``w:`` 开头：词级 n-gram；
        * ``c:`` 开头：字符级 n-gram。

        两者分开命名，避免同一个字符串在词空间和字符空间互相覆盖。

        算法细节：
        * 词 n-gram 用滑动窗口生成。例如 ``[a, b, c]`` 的 2-gram 为
          ``a b``、``b c``；高阶 n-gram 由 ``zip(*shifted)`` 生成，
          比嵌套循环更简洁，且完全由 C 层迭代器完成。
        * 字符 n-gram 在"去掉所有空白字符"的字符串上滑窗。
          去空白是为了让 ``"中国 高考"`` 与 ``"中国高考"`` 产生
          相同的字符特征，避免空格把相邻汉字切断。
        """
        if not isinstance(text, str):
            raise TypeError("待处理文本必须是 str")

        normalized = self._normalize_text(text)

        counts: dict[str, int] = {}
        need_tokens = bool(self.config.word_ngrams) or (
            bool(self.config.char_ngrams) and self.config.char_ngram_mode == "word_boundary"
        )
        tokens = self._tokenize(normalized) if need_tokens else []

        if self.config.word_ngrams:
            for n in self.config.word_ngrams:
                if n == 1:
                    for token in tokens:
                        key = _WORD_PREFIX + token
                        counts[key] = counts.get(key, 0) + 1
                elif len(tokens) >= n:
                    # zip(*[tokens[i:] for i in range(n)]) 等价于
                    # 对所有长度为 n 的连续窗口做笛卡尔遍历
                    for window in zip(*(tokens[i:] for i in range(n)), strict=False):
                        key = _WORD_PREFIX + _NGRAM_JOINER.join(window)
                        counts[key] = counts.get(key, 0) + 1

        if self.config.char_ngrams:
            if self.config.char_ngram_mode == "word_boundary":
                # 类似 sklearn 的 char_wb：只在词内部滑窗，并补首尾空格，
                # 避免英文 "new york" 生成跨词的 "w y" 之类噪声。
                for token in tokens:
                    padded = f" {token} "
                    for n in self.config.char_ngrams:
                        if len(padded) < n:
                            continue
                        for start in range(len(padded) - n + 1):
                            key = _CHAR_PREFIX + padded[start : start + n]
                            counts[key] = counts.get(key, 0) + 1
            else:
                # 全局模式：去掉所有空白后整串滑窗，中文标题效果最好。
                chars = "".join(normalized.split())
                for n in self.config.char_ngrams:
                    if len(chars) < n:
                        continue
                    for start in range(len(chars) - n + 1):
                        key = _CHAR_PREFIX + chars[start : start + n]
                        counts[key] = counts.get(key, 0) + 1

        return counts

    # ------------------------------------------------------------------
    # 2. fit：建词表 + 算 IDF
    # ------------------------------------------------------------------
    def fit(self, texts: Sequence[str]) -> GroceryTextConverter:
        """扫描训练语料，建立词表并计算 IDF。

        Args:
            texts: 训练文本序列（已按样本顺序排列）。

        Returns:
            ``self``，便于链式调用。

        Note:
            词表顺序经过 ``sorted`` 固定，保证同样的语料在任何机器上
            训练出的列号完全一致 —— 模型文件因此可跨平台复用。
        """
        doc_freq: dict[str, int] = {}
        n_docs = 0

        for text in texts:
            counts = self._extract_counts(text)
            n_docs += 1
            # 文档频率按"文档"计，同一文档内重复出现只算一次
            for key in counts:
                doc_freq[key] = doc_freq.get(key, 0) + 1

        if n_docs == 0:
            raise ValueError("训练语料为空，无法建立特征词表")

        effective_min_df = self._resolve_min_df(n_docs)
        items = [(key, df) for key, df in doc_freq.items() if df >= effective_min_df]

        if self.config.max_features is not None and len(items) > self.config.max_features:
            # 先按文档频率降序挑选，再按特征名排序固定列号。
            # 第二个排序键是特征名，用于在 DF 相同时保持确定性。
            items.sort(key=lambda pair: (-pair[1], pair[0]))
            items = items[: self.config.max_features]

        # 最终列号按特征名排序，保证跨平台、跨进程一致
        items.sort(key=lambda pair: pair[0])

        self.vocab = [key for key, _ in items]
        self.feature2idx = {key: idx for idx, key in enumerate(self.vocab)}
        self.idf = self._compute_idf(items, n_docs)
        self._fitted = True
        return self

    def _resolve_min_df(self, n_docs: int) -> int:
        """把 ``min_df`` 的"自动"配置解析成具体阈值。

        小语料（< 1000 篇）中，出现一次的特征往往就是唯一的类别线索，
        过滤掉会让模型几乎无特征可用，因此保留（阈值 1）。
        大语料中，出现一次的特征绝大多数是错别字、ID、偶然符号，
        过滤后既能降噪又能显著缩小矩阵（实测准确率反而更高）。
        """
        if self.config.min_df is not None:
            return self.config.min_df
        return 2 if n_docs >= 1000 else 1

    def _compute_idf(
        self,
        items: Sequence[tuple[str, int]],
        n_docs: int,
    ) -> np.ndarray | None:
        """计算 IDF 向量。

        公式：

        * 平滑模式（默认）：``idf = log((1 + N) / (1 + df)) + 1``
        * 非平滑模式：``idf = log(N / df)``

        加 1 的作用是保证即使某个特征在所有文档中都出现，
        权重也不会变成 0（否则该特征被完全抹掉）。
        """
        if not self.config.use_idf:
            return None

        idf = np.ones(len(items), dtype=np.float32)
        for idx, (_, df) in enumerate(items):
            if self.config.smooth_idf:
                idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            else:
                idf[idx] = math.log(n_docs / df)
        return idf

    # ------------------------------------------------------------------
    # 3. transform：生成稀疏向量
    # ------------------------------------------------------------------
    def transform(self, text: str) -> tuple[list[int], list[float]]:
        """把单条文本转换为 ``(列号列表, 取值列表)``。

        取值计算顺序：TF -> IDF -> L2 归一化。
        未在训练词表中出现过的特征直接忽略（这就是"未登录词"的代价，
        也是字符 n-gram 能缓解它的原因）。
        """
        if not self._fitted:
            raise RuntimeError("GroceryTextConverter 尚未 fit，无法 transform")

        counts = self._extract_counts(text)

        cols: list[int] = []
        values: list[float] = []
        for key, count in counts.items():
            idx = self.feature2idx.get(key)
            if idx is None:
                continue
            cols.append(idx)
            # 二值特征优先：只要出现过就记 1，忽略出现次数
            if self.config.binary:
                values.append(1.0)
            elif self.config.sublinear_tf:
                values.append(1.0 + math.log(count))
            else:
                values.append(float(count))

        if not cols:
            return [], []

        vector = np.asarray(values, dtype=np.float32)
        if self.config.use_idf and self.idf is not None:
            vector *= self.idf[np.asarray(cols, dtype=np.int64)]

        if self.config.norm == "l2":
            norm = float(np.sqrt(np.dot(vector, vector)))
            if norm > 0.0:
                vector /= norm

        # 按列号排序，让稀疏矩阵结构稳定（利于缓存与调试）
        order = np.argsort(np.asarray(cols, dtype=np.int64), kind="stable")
        sorted_cols = [cols[i] for i in order]
        sorted_values = vector[order].tolist()
        return sorted_cols, sorted_values

    def transform_corpus(self, texts: Sequence[str]) -> SparseMatrix:
        """批量转换，返回 :class:`SparseMatrix`。

        采用"先收集 Python 列表、最后一次性转 numpy"的方式，
        比逐条 ``np.append`` 快一个数量级。
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for row_idx, text in enumerate(texts):
            doc_cols, doc_values = self.transform(text)
            if not doc_cols:
                continue
            rows.extend([row_idx] * len(doc_cols))
            cols.extend(doc_cols)
            data.extend(doc_values)

        return SparseMatrix(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            data=np.asarray(data, dtype=np.float32),
            shape=(len(texts), len(self.vocab)),
        )

    # ------------------------------------------------------------------
    # 4. 持久化
    # ------------------------------------------------------------------
    def save(self, dest_dir: str | os.PathLike[str]) -> None:
        """把词表与 IDF 保存到目录。

        保存内容：

        * ``vocab.json``：特征名列表（UTF-8，可读、可审计）；
        * ``vectorizer.npz``：IDF 向量（numpy 压缩格式，跨平台）。

        不使用 pickle，避免 Python 版本差异和反序列化安全风险；
        词表用 JSON 也让模型文件可以被人直接打开检查。
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        with (dest / "vocab.json").open("w", encoding="utf-8") as fout:
            json.dump(self.vocab, fout, ensure_ascii=False)

        idf = self.idf if self.idf is not None else np.empty(0, dtype=np.float32)
        np.savez_compressed(dest / "vectorizer.npz", idf=idf)

    def load(
        self,
        src_dir: str | os.PathLike[str],
        config: FeatureConfig | None = None,
    ) -> GroceryTextConverter:
        """从目录恢复词表与 IDF。

        Args:
            src_dir: :meth:`save` 写入的目录。
            config: 训练时使用的特征配置；必须与保存时一致，
                否则特征名虽然能对上，但提取逻辑可能不同。

        Returns:
            ``self``。
        """
        src = Path(src_dir)
        with (src / "vocab.json").open("r", encoding="utf-8") as fin:
            self.vocab = json.load(fin)

        if config is not None:
            self.config = config
        # 直接调用 load() 时也要恢复停用词与用户词典，
        # 不能只依赖构造阶段的注册。
        self._stopwords = frozenset(self.config.stopwords)
        self._register_user_words()

        vectorizer_file = src / "vectorizer.npz"
        with np.load(vectorizer_file) as payload:
            idf = payload["idf"]

        self.idf = idf if idf.size else None
        self.feature2idx = {key: idx for idx, key in enumerate(self.vocab)}
        self._fitted = True
        return self

    def __repr__(self) -> str:
        return f"GroceryTextConverter(n_features={len(self.vocab)}, config={self.config})"
