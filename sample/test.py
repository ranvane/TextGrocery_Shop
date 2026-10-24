"""批量测试示例：训练后计算准确率、精确率与召回率。

运行方式（在仓库根目录）：

    python sample/test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgrocery import Grocery  # noqa: E402

TRAIN_SRC = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("education", "考生必读：新托福写作考试评分标准"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
    ("sports", "意甲首轮补赛交战记录 米兰客场8战不败国米10年连胜"),
]

TEST_SRC = [
    ("education", "福建春季公务员考试报名18日截止 2月6日考试"),
    ("sports", "意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜"),
]


def main() -> None:
    grocery = Grocery("sample")
    grocery.train(TRAIN_SRC)

    result = grocery.test(TEST_SRC)
    print("整体准确率:", result.accuracy_overall)
    print("逐标签精确率:", result.accuracy_labels)
    print("逐标签召回率:", result.recall_labels)
    print(result.to_table())


if __name__ == "__main__":
    main()
