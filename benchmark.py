"""TextGrocery_Shop 分类效果与速度基准脚本。

用途：在**你自己的数据**上对比不同特征配置，验证"字符 n-gram + TF-IDF"
是否真的带来提升。脚本不会修改库本身，也不上传任何数据。

数据格式（每行一条，UTF-8）：

* ``文本<TAB>标签``（默认，``--label-column last``）
* ``标签<TAB>文本``（``--label-column first``）

典型用法：

.. code-block:: bash

    # 默认三种配置对比
    python benchmark.py --train cnews.train.txt --test cnews.test.txt

    # 模拟"超短文本"：把每条文本截断到前 10 个字符
    python benchmark.py --train cnews.train.txt --test cnews.test.txt --truncate 10

    # 只跑指定配置，限制样本量以加快速度
    python benchmark.py --train cnews.train.txt --test cnews.test.txt \
        --configs legacy,word_char --train-size 30000 --test-size 5000

输出的指标：

* ``accuracy``：整体准确率；
* ``train_s``：训练耗时（含分词与特征构建）；
* ``predict_s``：测试集预测耗时；
* ``features``：词表大小。

注意：不同数据集的绝对准确率不可直接比较，只有**同一数据集上不同配置**
之间的差异才有意义。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tgrocery import Grocery

CONFIGS = {
    # 尽量贴近旧版 TextGrocery：词 1-2gram、二值特征、L2 归一化、无 IDF
    "legacy": dict(
        word_ngrams=(1, 2),
        char_ngrams=(),
        binary=True,
        sublinear_tf=False,
        use_idf=False,
        norm="l2",
    ),
    # 旧版特征 + 现代 TF-IDF 加权（不含字符 n-gram）
    "word_tfidf": dict(
        word_ngrams=(1, 2),
        char_ngrams=(),
        binary=False,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
    ),
    # 默认推荐配置：词 + 字符 n-gram + TF-IDF
    "word_char": dict(
        word_ngrams=(1, 2),
        char_ngrams=(2, 3),
        binary=False,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
    ),
}


def load_dataset(
    path: str | Path,
    label_column: str = "last",
    delimiter: str = "\t",
    limit: int | None = None,
    truncate: int | None = None,
) -> list[tuple[str, str]]:
    """读取数据集并返回 ``(标签, 文本)`` 列表。

    Args:
        path: 数据文件路径（UTF-8）。
        label_column: ``"last"`` 或 ``"first"``。
        delimiter: 列分隔符。
        limit: 最多读取多少条（``None`` 表示全部）。
        truncate: 若给出，则把文本截断到前 N 个字符，用于模拟超短文本。
    """
    samples: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue
            if label_column == "first":
                label, text = parts[0], delimiter.join(parts[1:])
            else:
                text, label = delimiter.join(parts[:-1]), parts[-1]
            if truncate is not None:
                text = text[:truncate]
            samples.append((label.strip(), text))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def evaluate(
    config_name: str,
    options: dict,
    train_samples: list[tuple[str, str]],
    test_samples: list[tuple[str, str]],
    C: float,
    max_iter: int,
    min_df: int,
) -> dict:
    """训练并评估一个配置，返回指标字典。"""
    started = time.perf_counter()
    grocery = Grocery(
        f"benchmark_{config_name}",
        C=C,
        max_iter=max_iter,
        min_df=min_df,
        **options,
    )
    grocery.train(train_samples)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = grocery.test(test_samples)
    predict_seconds = time.perf_counter() - started

    return {
        "config": config_name,
        "accuracy": result.accuracy_overall,
        "train_s": train_seconds,
        "predict_s": predict_seconds,
        "features": len(grocery.model.text_converter.vocab),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TextGrocery_Shop 基准测试")
    parser.add_argument("--train", required=True, help="训练集文件")
    parser.add_argument("--test", required=True, help="测试集文件")
    parser.add_argument(
        "--configs",
        default="legacy,word_tfidf,word_char",
        help="逗号分隔的配置名，可选：legacy,word_tfidf,word_char",
    )
    parser.add_argument("--label-column", choices=["first", "last"], default="last")
    parser.add_argument("--train-size", type=int, default=None, help="训练集最多读取条数")
    parser.add_argument("--test-size", type=int, default=None, help="测试集最多读取条数")
    parser.add_argument("--truncate", type=int, default=None, help="把文本截断到前 N 个字符")
    parser.add_argument("--C", type=float, default=8.0, help="正则化强度倒数")
    parser.add_argument("--min-df", type=int, default=None, help="特征最小文档频率（默认自动）")
    parser.add_argument("--max-iter", type=int, default=50, help="每类最大迭代次数")
    args = parser.parse_args()

    train_samples = load_dataset(
        args.train, args.label_column, limit=args.train_size, truncate=args.truncate
    )
    test_samples = load_dataset(
        args.test, args.label_column, limit=args.test_size, truncate=args.truncate
    )
    if not train_samples or not test_samples:
        raise SystemExit("训练集或测试集为空，请检查路径与格式")

    print(
        f"train={len(train_samples)} test={len(test_samples)} "
        f"C={args.C} min_df={args.min_df} truncate={args.truncate}"
    )

    rows = []
    for config_name in [name.strip() for name in args.configs.split(",") if name.strip()]:
        if config_name not in CONFIGS:
            raise SystemExit(f"未知配置: {config_name}")
        rows.append(
            evaluate(
                config_name,
                CONFIGS[config_name],
                train_samples,
                test_samples,
                C=args.C,
                max_iter=args.max_iter,
                min_df=args.min_df,
            )
        )

    header = f"{'config':<12}{'accuracy':>10}{'train_s':>10}{'predict_s':>11}{'features':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['config']:<12}"
            f"{row['accuracy'] * 100:>9.2f}%"
            f"{row['train_s']:>10.2f}"
            f"{row['predict_s']:>11.2f}"
            f"{row['features']:>10d}"
        )


if __name__ == "__main__":
    main()
