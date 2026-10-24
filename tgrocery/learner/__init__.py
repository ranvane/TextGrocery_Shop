"""训练器子包。

对外只暴露纯 numpy 的线性模型实现：

* :class:`tgrocery.learner.linear.LinearClassifier`
* :func:`tgrocery.learner.linear.lbfgs_minimize`

历史上依赖 C 版 LibLinear 的 ``learner.py``、``util.c`` 与
``liblinear/`` 已在 0.2.x 中移除，当前只保留纯 numpy 实现。
这样 ``pip install`` 不需要任何 C 编译器，Windows / Linux / macOS
行为一致；需要对照旧实现时可查看上游仓库的历史版本。
"""

from .linear import LinearClassifier, lbfgs_minimize

__all__ = ["LinearClassifier", "lbfgs_minimize"]
