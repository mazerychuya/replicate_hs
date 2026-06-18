# -*- coding: utf-8 -*-
"""
verify_against_imodels.py
==========================

【可选脚本，需要联网环境运行】

这是"自己重写算法 + 用官方代码交叉验证"这个复现思路里的第二步。
本沙箱环境无法访问网络，所以这一步请你在自己的电脑上跑：

    pip install imodels

跑完后，会把我们自己写的 HSTreeClassifier / HSTreeRegressor (hs_lib.py)
和作者官方包 imodels.HSTreeClassifier / imodels.HSTreeRegressor
在完全相同的数据、完全相同的 lambda 下的预测结果做逐样本比较。

如果两者输出几乎完全一致（数值误差在 1e-8 量级，属于浮点误差），
就证明我们自己实现的算法和论文作者的官方实现是等价的——
这是给复现报告"代码"部分加分的一个很好的证据。
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from hs_lib import HSTreeClassifier, HSTreeRegressor

try:
    import imodels
except ImportError:
    raise SystemExit(
        "未检测到 imodels 包。请先在有网络的环境运行: pip install imodels\n"
        "（本沙箱环境无网络，这个脚本需要你在自己电脑上跑）"
    )


def compare_classifier():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    lam = 10.0
    max_leaf_nodes = 16

    ours = HSTreeClassifier(
        estimator=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0),
        lam=lam,
    ).fit(Xtr, ytr)

    theirs = imodels.HSTreeClassifier(
        estimator_=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0),
        reg_param=lam,
    )
    theirs.fit(Xtr, ytr)

    p_ours = ours.predict_proba(Xte)[:, 1]
    p_theirs = theirs.predict_proba(Xte)[:, 1]
    max_diff = np.max(np.abs(p_ours - p_theirs))
    print(f"[分类] breast_cancer, lambda={lam}, max_leaf_nodes={max_leaf_nodes}")
    print(f"  自己实现 vs imodels 官方实现，预测概率最大差异: {max_diff:.2e}")
    print("  -> 若差异在 1e-6 以下，可认为两种实现数学等价。")


def compare_regressor():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    lam = 10.0
    max_leaf_nodes = 16

    ours = HSTreeRegressor(
        estimator=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0),
        lam=lam,
    ).fit(Xtr, ytr)

    theirs = imodels.HSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0),
        reg_param=lam,
    )
    theirs.fit(Xtr, ytr)

    pred_ours = ours.predict(Xte)
    pred_theirs = theirs.predict(Xte)
    max_diff = np.max(np.abs(pred_ours - pred_theirs))
    print(f"\n[回归] diabetes, lambda={lam}, max_leaf_nodes={max_leaf_nodes}")
    print(f"  自己实现 vs imodels 官方实现，预测值最大差异: {max_diff:.4f}")
    print("  -> 若差异远小于目标变量的量级，可认为两种实现等价。")


if __name__ == "__main__":
    compare_classifier()
    compare_regressor()