"""自定义分词器示例。

``custom_tokenize`` 接收一段文本，返回 token 可迭代对象。
本例使用 Python 内置的 ``list``，即"逐字符切分"——
对中文短文本来说，字符级特征本身就是一个很强的基线。

运行方式（在仓库根目录）：

    python sample/custom_tokenize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgrocery import Grocery  # noqa: E402

TRAIN_SRC = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
]


def char_tokenize(text: str) -> list[str]:
    """逐字符切分（空格除外）。"""
    return [char for char in text if not char.isspace()]


def main() -> None:
    custom_grocery = Grocery("custom", custom_tokenize=char_tokenize)
    custom_grocery.train(TRAIN_SRC)
    print("预测结果:", custom_grocery.predict("考生必读：新托福写作考试评分标准"))


if __name__ == "__main__":
    main()
