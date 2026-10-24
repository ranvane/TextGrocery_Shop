"""纯 numpy 的线性分类器：一对多（OvR）逻辑回归 + L-BFGS 优化。

## 为什么不再依赖 C 版 LibLinear

原版 TextGrocery 通过 ``ctypes`` 加载 ``util.so.1`` / ``liblinear.so.1``，
这两个动态库需要在目标机器上编译。结果是：

* Windows 上默认完全不可用（``setup.py`` 直接调用 ``make`` 和 ``cp``）；
* 即使能编译，Python 版本升级、编译器差异、ARM 平台都会带来额外麻烦；
* 用户 ``pip install`` 后还要排错编译环境，违背"简单"的初衷。

本模块用 numpy 重新实现了训练所需的核心运算，做到"安装即用"。
关键是用 :func:`numpy.bincount` 完成稀疏矩阵乘向量：

* ``X @ w``：``bincount(rows, weights=data * w[cols], minlength=n_rows)``
* ``X.T @ v``：``bincount(cols, weights=data * v[rows], minlength=n_features)``

``bincount`` 的循环在 C 层完成，因此虽然是纯 Python 接口，
矩阵运算本身依然很快，且不需要任何编译步骤。

## 算法原理

**一对多（One-vs-Rest）逻辑回归**

对每个类别 :math:`k`，训练一个二分类器，把"属于 k"视为正类、
其余视为负类。第 :math:`k` 个分类器的目标是：

.. math::

    \\min_{w_k, b_k} \\; \\frac{1}{2}\\lVert w_k \\rVert^2
    + C \\sum_{i=1}^{n} s_i \\log\\!\\left(
        1 + \\exp\\!\\left(-y_{ik}(w_k^\\top x_i + b_k)\\right)
    \\right)

其中 :math:`y_{ik} \\in \\{-1, +1\\}`，:math:`s_i` 是样本权重
（``class_weight="balanced"`` 时用于缓解类别不平衡）。

预测时取所有二分类器中得分最高者：

.. math::

    \\hat{y} = \\arg\\max_k \\; (w_k^\\top x + b_k)

**为什么选逻辑回归而不是线性 SVM**

* 逻辑回归的损失是光滑凸函数，可以直接用 L-BFGS 这类拟牛顿法，
  收敛快、无需调学习率；
* 输出是 logit，经 softmax 后可解释为概率，比 SVM 的间隔值更友好；
* 在短文本 TF-IDF 特征上，逻辑回归与线性 SVM 的准确率通常在同一水平，
  但逻辑回归更不容易被离群点主导。

**L-BFGS（有限内存 BFGS）**

BFGS 用近似 Hessian 矩阵决定下降方向，但存储 :math:`O(d^2)` 不可行
（本场景 :math:`d` 可达几十万）。L-BFGS 只保留最近 ``m`` 步的
位移差 :math:`s_i` 与梯度差 :math:`y_i`，通过"两循环递归"
（two-loop recursion）隐式地算出 :math:`H_k g_k`，存储降为 :math:`O(md)`。
配合 Armijo 回溯线搜索，即可稳定地最小化上面的凸目标。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..converter import SparseMatrix

__all__ = ["LinearClassifier", "lbfgs_minimize"]


# 目标函数类型：输入参数向量，返回 (损失值, 梯度向量)
Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。

    直接写 ``1 / (1 + exp(-z))`` 在 ``z`` 很负时会溢出；
    这里按符号分支，把指数项限制在 ``exp(-|z|)`` 范围内。
    """
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _two_loop_recursion(
    grad: np.ndarray,
    s_history: list[np.ndarray],
    y_history: list[np.ndarray],
) -> np.ndarray:
    """L-BFGS 两循环递归，返回近似 ``-H * grad``（即下降方向）。

    记号与 Nocedal & Wright 的算法 7.4 一致：

    * 第一轮从最新历史向旧历史回退，累计缩放系数 ``alpha``；
    * 用最后一次的曲率估计 ``gamma = sᵀy / yᵀy`` 作为初始 Hessian 缩放；
    * 第二轮正向回放，修正方向向量。
    """
    q = grad.copy()
    alphas: list[tuple[float, float]] = []

    for s_vec, y_vec in zip(reversed(s_history), reversed(y_history), strict=True):
        curvature = float(np.dot(y_vec, s_vec))
        if curvature == 0.0:
            # 曲率为 0 表示该步没有提供有效曲率信息。
            # 仍然记录一对 0，保证两轮循环中历史向量严格对齐。
            alphas.append((0.0, 0.0))
            continue
        rho = 1.0 / curvature
        alpha = rho * float(np.dot(s_vec, q))
        q -= alpha * y_vec
        alphas.append((rho, alpha))

    if s_history:
        s_last, y_last = s_history[-1], y_history[-1]
        denominator = float(np.dot(y_last, y_last))
        gamma = float(np.dot(s_last, y_last)) / denominator if denominator > 0 else 1.0
        if gamma <= 0.0:
            gamma = 1.0
    else:
        gamma = 1.0

    r = gamma * q

    for (rho, alpha), s_vec, y_vec in zip(reversed(alphas), s_history, y_history, strict=True):
        beta = rho * float(np.dot(y_vec, r))
        r += s_vec * (alpha - beta)

    return -r


def lbfgs_minimize(
    objective: Objective,
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    memory: int = 10,
    max_line_search: int = 30,
) -> tuple[np.ndarray, int]:
    """用 L-BFGS + Armijo 回溯线搜索最小化光滑凸函数。

    Args:
        objective: 目标函数，输入参数向量，返回 ``(loss, gradient)``。
        x0: 初始参数（通常为全零）。
        max_iter: 最大迭代次数。
        tol: 收敛阈值，梯度无穷范数小于该值即停止。
        memory: 保留的历史步数 ``m``，常用 5~20。
        max_line_search: 单次线搜索最多减半步数。

    Returns:
        ``(最优参数, 实际迭代次数)``。

    Note:
        线搜索采用 Armijo 条件
        ``f(x + t d) <= f(x) + c1 * t * gᵀd``，其中 ``c1 = 1e-4``。
        当方向不是下降方向（数值误差导致）时，直接退回最速下降方向。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    value, grad = objective(x)
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        if float(np.max(np.abs(grad))) <= tol:
            break

        direction = _two_loop_recursion(grad, s_history, y_history)
        directional_derivative = float(np.dot(grad, direction))

        if directional_derivative >= 0.0:
            # 非下降方向：清空历史，退回负梯度方向
            direction = -grad
            directional_derivative = -float(np.dot(grad, grad))
            s_history.clear()
            y_history.clear()

        step = 1.0
        accepted = False
        new_x = x
        new_value = value
        new_grad = grad

        for _ in range(max_line_search):
            candidate = x + step * direction
            candidate_value, candidate_grad = objective(candidate)
            if np.isfinite(candidate_value) and candidate_value <= (
                value + 1e-4 * step * directional_derivative
            ):
                new_x, new_value, new_grad = candidate, candidate_value, candidate_grad
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # 线搜索失败：说明已经到达数值精度极限，停止迭代
            break

        s_vec = new_x - x
        y_vec = new_grad - grad
        curvature = float(np.dot(s_vec, y_vec))
        if curvature > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            if len(s_history) > memory:
                s_history.pop(0)
                y_history.pop(0)

        previous_value = value
        x, value, grad = new_x, new_value, new_grad

        # 损失几乎不再下降时提前停止，避免空转
        if abs(previous_value - value) <= 1e-10 * max(1.0, abs(previous_value)):
            break

    return x, iterations


@dataclass
class LinearClassifier:
    """多分类线性模型（OvR 逻辑回归）。

    Attributes:
        coef: 权重矩阵，形状 ``(n_classes, n_features)``。
        intercept: 偏置向量，形状 ``(n_classes,)``。
        labels: 类别名列表，``coef`` 的第 ``k`` 行对应 ``labels[k]``。
    """

    coef: np.ndarray
    intercept: np.ndarray
    labels: list[str]

    @property
    def n_classes(self) -> int:
        return int(self.coef.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.coef.shape[1])

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        features: SparseMatrix,
        y: np.ndarray,
        labels: Sequence[str],
        C: float = 1.0,
        class_weight: str | dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> LinearClassifier:
        """训练 OvR 逻辑回归模型。

        Args:
            features: 训练样本的稀疏特征矩阵。
            y: 每个样本的类别下标（``0 .. n_classes-1``）。
            labels: 类别名列表，下标与 ``y`` 对应。
            C: 正则化强度的倒数。``C`` 越大正则越弱。
                短文本 TF-IDF 特征常用 ``1.0 ~ 10.0``。
            class_weight: ``None``、``"balanced"`` 或
                ``{类别名: 权重}``。类别不平衡时建议用 ``"balanced"``。
            max_iter: 每个二分类器的 L-BFGS 最大迭代次数。
            tol: 梯度收敛阈值。
            verbose: 是否打印每个类别的训练信息。

        Returns:
            训练好的 :class:`LinearClassifier`。
        """
        labels = [str(label) for label in labels]
        n_classes = len(labels)
        n_features = features.shape[1]
        n_samples = features.shape[0]

        coef = np.zeros((n_classes, n_features), dtype=np.float64)
        intercept = np.zeros(n_classes, dtype=np.float64)

        if n_classes == 1:
            # 只有一个类别时没有可学习的内容，直接返回常量分类器
            return cls(
                coef=coef.astype(np.float32),
                intercept=intercept.astype(np.float32),
                labels=labels,
            )

        # 预先取出稀疏数组，闭包内直接引用，减少属性查找
        rows = features.rows
        cols = features.cols
        data = features.data.astype(np.float64, copy=False)
        reg_lambda = 1.0 / float(C)

        base_weights = _resolve_class_weights(class_weight, y, labels)

        for class_idx in range(n_classes):
            target = (y == class_idx).astype(np.float64)
            sample_weight = _binary_sample_weights(base_weights, class_idx, target)

            def objective(w: np.ndarray, target=target, sample_weight=sample_weight):
                # X @ w   （使用 bincount 在 C 层累加）
                scores = np.bincount(rows, weights=data * w[cols], minlength=n_samples)
                scores += w[-1]  # 偏置项

                # 逻辑损失：log(1 + exp(z)) - y*z，用 logaddexp 防止溢出
                loss = float(np.dot(sample_weight, np.logaddexp(0.0, scores) - target * scores))
                # L2 正则只作用于权重，不惩罚偏置
                loss += 0.5 * reg_lambda * float(np.dot(w[:-1], w[:-1]))

                residual = (_sigmoid(scores) - target) * sample_weight
                # X.T @ residual
                grad = np.bincount(cols, weights=data * residual[rows], minlength=n_features)
                grad_full = np.empty(n_features + 1, dtype=np.float64)
                grad_full[:-1] = grad + reg_lambda * w[:-1]
                grad_full[-1] = float(np.sum(residual))
                return loss, grad_full

            w0 = np.zeros(n_features + 1, dtype=np.float64)
            w, used_iterations = lbfgs_minimize(objective, w0, max_iter=max_iter, tol=tol)
            coef[class_idx] = w[:-1]
            intercept[class_idx] = w[-1]

            if verbose:
                print(f"[tgrocery] class={labels[class_idx]!r} iterations={used_iterations}")

        # 用 float32 存储可将模型体积减半，预测精度损失可以忽略
        return cls(
            coef=coef.astype(np.float32),
            intercept=intercept.astype(np.float32),
            labels=labels,
        )

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def decision_function(
        self,
        cols: Sequence[int],
        values: Sequence[float],
    ) -> np.ndarray:
        """计算一条样本在各类别上的原始得分（logit）。

        稀疏点积只需取权重矩阵对应的列：
        ``scores = coef[:, cols] @ values + intercept``。
        该运算的复杂度是 O(类别数 × 非零特征数)，与特征维度无关，
        因此预测速度很快。
        """
        # 注意：cols 可能是 numpy 数组，不能直接用 ``if not cols`` 判断，
        # 否则多元素数组的真值会抛出 ValueError。这里只判断长度。
        if len(cols) == 0:
            return self.intercept.astype(np.float64, copy=True)
        col_index = np.asarray(cols, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float64)
        return self.coef[:, col_index] @ value_array + self.intercept

    def predict_one(self, cols: Sequence[int], values: Sequence[float]) -> tuple[str, np.ndarray]:
        """返回 ``(预测标签, 全部类别的得分)``。"""
        scores = self.decision_function(cols, values)
        return self.labels[int(np.argmax(scores))], scores

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, dest_dir: str | os.PathLike[str]) -> None:
        """把权重与偏置保存为 ``model.npz``。

        类别名保存在上层 ``metadata.json`` 中，这里只存数值数组，
        避免使用允许任意对象反序列化的 ``pickle``。
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest / "model.npz",
            coef=self.coef.astype(np.float32),
            intercept=self.intercept.astype(np.float32),
        )

    @classmethod
    def load(
        cls,
        src_dir: str | os.PathLike[str],
        labels: Sequence[str],
    ) -> LinearClassifier:
        """从 ``model.npz`` 恢复模型。"""
        with np.load(Path(src_dir) / "model.npz") as payload:
            coef = payload["coef"].astype(np.float32)
            intercept = payload["intercept"].astype(np.float32)
        return cls(coef=coef, intercept=intercept, labels=[str(label) for label in labels])


def _resolve_class_weights(
    class_weight: str | dict[str, float] | None,
    y: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray | None:
    """把 ``class_weight`` 参数解析成"每个类别一个权重"的数组。

    支持三种形式：

    * ``None``：所有类别权重为 1；
    * ``"balanced"``：``weight_k = n_samples / (n_classes * count_k)``，
      让每个类别在损失函数中的总贡献大致相等；
    * 字典：按类别名指定权重，未指定的类别默认为 1。
    """
    n_classes = len(labels)
    n_samples = len(y)

    if class_weight is None:
        return np.ones(n_classes, dtype=np.float64)

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight 只支持 None、'balanced' 或字典")
        weights = np.ones(n_classes, dtype=np.float64)
        for class_idx in range(n_classes):
            count = int(np.sum(y == class_idx))
            weights[class_idx] = n_samples / (n_classes * count) if count > 0 else 0.0
        return weights

    if isinstance(class_weight, dict):
        weights = np.ones(n_classes, dtype=np.float64)
        for class_idx, label in enumerate(labels):
            if label in class_weight:
                weights[class_idx] = float(class_weight[label])
        return weights

    raise TypeError("class_weight 类型不支持")


def _binary_sample_weights(
    class_weights: np.ndarray | None,
    class_idx: int,
    target: np.ndarray,
) -> np.ndarray:
    """把多分类权重转换成当前二分类问题的逐样本权重。

    对第 ``k`` 个二分类器：正类样本（属于 k）使用 ``weight_k``，
    负类样本取其余类别的平均值。这样既保留了类别权重信息，
    又不会让负类权重随类别数剧烈放大。
    """
    if class_weights is None:
        return np.ones_like(target)

    n_classes = len(class_weights)
    if n_classes <= 1:
        return np.ones_like(target)

    others = [class_weights[i] for i in range(n_classes) if i != class_idx]
    negative_weight = float(np.mean(others)) if others else 1.0
    return np.where(target > 0.5, class_weights[class_idx], negative_weight)
