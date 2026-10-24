"""基础示例：训练 + 预测。

运行方式（在仓库根目录）：

    python sample/classify.py
"""

import sys
from pathlib import Path

# 允许直接以脚本方式运行示例，而无需先 pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgrocery import Grocery  # noqa: E402

TRAIN_SRC = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
]


def main() -> None:
    grocery = Grocery("sample")
    grocery.train(TRAIN_SRC)
    print("模型是否可用:", grocery.get_load_status())

    result = grocery.predict("考生必读：新托福写作考试评分标准")
    print("预测标签:", result)
    print("决策值:", result.dec_values)
    print("概率:", result.probabilities)


if __name__ == "__main__":
    main()
