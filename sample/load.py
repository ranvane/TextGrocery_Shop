"""保存 / 加载示例。

运行方式（在仓库根目录）：

    python sample/load.py

模型默认保存在系统临时目录中，脚本结束自动清理。
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgrocery import Grocery  # noqa: E402

TRAIN_SRC = [
    ("education", "名师指导托福语法技巧：名词的复数形式"),
    ("education", "中国高考成绩海外认可 是狼来了吗"),
    ("sports", "图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼"),
    ("sports", "四川丹棱举行全国长距登山挑战赛 近万人参与"),
]


def main() -> None:
    original_dir = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            grocery = Grocery("sample")
            grocery.train(TRAIN_SRC)
            grocery.save()
            print("已保存文件:", sorted(os.listdir("sample")))

            new_grocery = Grocery("sample")
            new_grocery.load()
            print("加载后预测:", new_grocery.predict("考生必读：新托福写作考试评分标准"))
        finally:
            # Windows 下不能删除仍被当作工作目录的文件夹，
            # 因此离开临时目录后再让 TemporaryDirectory 清理。
            os.chdir(original_dir)


if __name__ == "__main__":
    main()
