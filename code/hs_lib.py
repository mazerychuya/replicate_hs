# -*- coding: utf-8 -*-
"""
hs_lib.py
=========

从零复现论文核心算法:

    Agarwal, A., Tan, Y. S., Ronen, O., Singh, C., & Yu, B. (2022).
    "Hierarchical Shrinkage: Improving the Accuracy and Interpretability
    of Tree-Based Models."
    Proceedings of the 39th International Conference on Machine Learning
    (ICML 2022), PMLR 162:111-135.  arXiv:2202.00858

核心公式 (论文 Eq. 1, Hierarchical Shrinkage, HS):

    设查询点 x 落入叶子节点 t_L, 其从根到叶的路径为 t_0(根) -> t_1 -> ... -> t_L(叶)。
    N(t)  : 节点 t 包含的训练样本数（带权）。
    E_t{y}: 节点 t 上训练响应的均值（回归）或类别比例（分类）。

    原始树模型的预测可以写成"望远镜求和" (telescoping sum)：
        f(x) = E_{t0}{y} + sum_{l=1}^{L} ( E_{tl}{y} - E_{t(l-1)}{y} )

    HS 把每一项按照其【父节点】样本数 N(t_{l-1}) 做收缩：
        f_lambda(x) = E_{t0}{y} + sum_{l=1}^{L}
                        ( E_{tl}{y} - E_{t(l-1)}{y} ) / ( 1 + lambda / N(t_{l-1}) )      ... (1)

    这等价于递归定义：
        hs(root)  = raw(root)
        hs(node)  = hs(parent) + ( raw(node) - raw(parent) ) * 1/(1+lambda/N(parent))

论文中还给出了一个用于对比的朴素方法 leaf-based shrinkage (LBS, Eq. 2,
类似 XGBoost / BART 中使用的收缩方式)，只收缩叶子到根这一步：

        f^l_lambda(x) = E_{t0}{y} + ( E_{tL}{y} - E_{t0}{y} ) / ( 1 + lambda / N(tL) )   ... (2)

本文件实现了 HS 和 LBS 两种方法，并提供与 scikit-learn 兼容的
Classifier / Regressor 包装类（包括内置交叉验证选择 lambda 的 *CV 版本），
可直接套在任意 sklearn 的树模型（DecisionTree*, RandomForest*,
ExtraTrees*, GradientBoosting*）外面使用。
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score


# ----------------------------------------------------------------------
# 1. 核心算法：对单棵树的 tree_ 结构计算每个节点的收缩后取值
# ----------------------------------------------------------------------

def _compute_node_values(tree_, lam, is_classifier, method="hs"):
    """对一棵已经 fit 好的 sklearn 树 (tree_ = estimator.tree_) 计算
    每个节点收缩后的预测值，数组形状与 tree_.value 完全一致，
    因此可以直接用 estimator.apply(X) 得到的叶子节点 id 做索引。

    method: "hs"  -> Hierarchical Shrinkage (论文 Eq.1, 本文复现的核心方法)
            "lbs" -> Leaf-Based Shrinkage   (论文 Eq.2, 对比基线)
            "none"-> 不收缩，直接返回原始值（等价于普通 CART/RF）
    """
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    n_samples = tree_.weighted_n_node_samples  # N(t)

    raw = tree_.value.astype(float).copy()  # shape (n_nodes, n_outputs, K)

    if is_classifier:
        # sklearn 分类树的 value 存的是（带权）类别计数，需要归一化为
        # 概率，对应论文里的 E_t{y}（这里 y 是 one-hot 的类别指示变量）
        sums = raw.sum(axis=2, keepdims=True)
        sums[sums == 0] = 1.0
        raw = raw / sums

    if method == "none":
        return raw

    out = np.zeros_like(raw)

    if method == "lbs":
        # 公式(2)：只用根节点和叶子节点，不管中间路径
        root_val = raw[0]
        shrink = 1.0 / (1.0 + lam / np.maximum(n_samples, 1e-12))
        for node in range(n_nodes):
            out[node] = root_val + (raw[node] - root_val) * shrink[node]
        return out

    # method == "hs"：公式(1)，沿着根->叶路径递归累加
    # 用栈做非递归遍历，避免深树超出 Python 递归深度限制
    stack = [(0, -1)]
    while stack:
        node, parent = stack.pop()
        if parent == -1:
            out[node] = raw[node]
        else:
            n_parent = max(n_samples[parent], 1e-12)
            shrink = 1.0 / (1.0 + lam / n_parent)
            out[node] = out[parent] + (raw[node] - raw[parent]) * shrink
        if children_left[node] != -1:
            stack.append((children_left[node], node))
            stack.append((children_right[node], node))
    return out


def _is_ensemble(estimator):
    return hasattr(estimator, "estimators_")


def _hs_predict_raw(estimator, node_values, X):
    """给定一个 fit 好的 estimator（单树或森林）以及对应的收缩后节点取值，
    返回对 X 的预测（分类返回概率矩阵，回归返回标量数组）。"""
    if _is_ensemble(estimator):
        acc = None
        for est, nv in zip(estimator.estimators_, node_values):
            leaf_idx = est.apply(X)
            contrib = nv[leaf_idx]  # shape (n_samples, n_outputs, K)
            acc = contrib if acc is None else acc + contrib
        acc = acc / len(estimator.estimators_)
        return acc
    else:
        leaf_idx = estimator.apply(X)
        return node_values[leaf_idx]


# ----------------------------------------------------------------------
# 2. sklearn 兼容的包装类
# ----------------------------------------------------------------------

class _HSBase(BaseEstimator):
    def __init__(self, estimator=None, lam=1.0, method="hs"):
        self.estimator = estimator
        self.lam = lam
        self.method = method  # "hs" / "lbs" / "none"

    def _base_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        raise ValueError("必须提供一个已配置好的 sklearn 树模型作为 estimator")

    def _fit_values(self, is_classifier):
        if _is_ensemble(self.estimator_):
            self.node_values_ = [
                _compute_node_values(est.tree_, self.lam, is_classifier, self.method)
                for est in self.estimator_.estimators_
            ]
        else:
            self.node_values_ = _compute_node_values(
                self.estimator_.tree_, self.lam, is_classifier, self.method
            )

    def set_lambda(self, lam):
        """post-hoc 方法的核心优势：换 lambda 不需要重新训练树，只需要
        重新做一次 O(节点数) 的收缩计算。"""
        self.lam = lam
        is_classifier = isinstance(self, ClassifierMixin)
        self._fit_values(is_classifier)
        return self


class HSTreeRegressor(_HSBase, RegressorMixin):
    """Hierarchical-Shrinkage 包装的回归器，可套在
    DecisionTreeRegressor / RandomForestRegressor / ExtraTreesRegressor 外。"""

    def fit(self, X, y):
        self.estimator_ = self._base_estimator().fit(X, y)
        self._fit_values(is_classifier=False)
        return self

    def predict(self, X):
        out = _hs_predict_raw(self.estimator_, self.node_values_, X)
        return out[:, 0, 0]


class HSTreeClassifier(_HSBase, ClassifierMixin):
    """Hierarchical-Shrinkage 包装的分类器，可套在
    DecisionTreeClassifier / RandomForestClassifier / ExtraTreesClassifier 外。"""

    def fit(self, X, y):
        self.estimator_ = self._base_estimator().fit(X, y)
        self.classes_ = self.estimator_.classes_
        self._fit_values(is_classifier=True)
        return self

    def predict_proba(self, X):
        probs = _hs_predict_raw(self.estimator_, self.node_values_, X)[:, 0, :]
        probs = np.clip(probs, 0, None)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probs / row_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ----------------------------------------------------------------------
# 3. 内置交叉验证选择 lambda 的版本（对应 imodels 中的 *CV 类）
# ----------------------------------------------------------------------

DEFAULT_LAMBDA_GRID = (0.1, 1, 2, 5, 10, 25, 50, 100, 200)


def _cv_select_lambda(estimator_template, X, y, is_classifier, method,
                       lambda_grid, n_splits=3, random_state=0):
    """K-fold 内部交叉验证选择 lambda。

    关键点（呼应论文里 HS 是 post-hoc 方法、速度快的说法）：
    每一折只需要训练 1 次基础树/森林，然后对网格中的每个 lambda
    只是重新做一次 O(节点数) 的收缩计算，而不是重新训练模型。
    """
    if is_classifier:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = {lam: [] for lam in lambda_grid}
    for train_idx, val_idx in splitter.split(X, y):
        base = clone(estimator_template).fit(X[train_idx], y[train_idx])
        if _is_ensemble(base):
            raw_values_per_tree = [est.tree_ for est in base.estimators_]
        else:
            raw_values_per_tree = [base.tree_]

        for lam in lambda_grid:
            if _is_ensemble(base):
                node_values = [
                    _compute_node_values(t, lam, is_classifier, method) for t in raw_values_per_tree
                ]
            else:
                node_values = _compute_node_values(raw_values_per_tree[0], lam, is_classifier, method)
            pred = _hs_predict_raw(base, node_values, X[val_idx])
            if is_classifier:
                proba = np.clip(pred[:, 0, :], 0, None)
                rs = proba.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                proba = proba / rs
                if proba.shape[1] == 2:
                    score = roc_auc_score(y[val_idx], proba[:, 1])
                else:
                    # 多分类用准确率代替 AUC
                    score = (np.argmax(proba, axis=1) == y[val_idx]).mean()
            else:
                score = r2_score(y[val_idx], pred[:, 0, 0])
            scores[lam].append(score)

    mean_scores = {lam: np.mean(v) for lam, v in scores.items()}
    best_lam = max(mean_scores, key=mean_scores.get)
    return best_lam, mean_scores


class HSTreeRegressorCV(HSTreeRegressor):
    def __init__(self, estimator=None, lambda_grid=DEFAULT_LAMBDA_GRID,
                 method="hs", n_splits=3, random_state=0):
        super().__init__(estimator=estimator, lam=1.0, method=method)
        self.lambda_grid = lambda_grid
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.best_lambda_, self.cv_scores_ = _cv_select_lambda(
            self._base_estimator(), X, y, is_classifier=False, method=self.method,
            lambda_grid=self.lambda_grid, n_splits=self.n_splits,
            random_state=self.random_state)
        self.lam = self.best_lambda_
        return super().fit(X, y)


class HSTreeClassifierCV(HSTreeClassifier):
    def __init__(self, estimator=None, lambda_grid=DEFAULT_LAMBDA_GRID,
                 method="hs", n_splits=3, random_state=0):
        super().__init__(estimator=estimator, lam=1.0, method=method)
        self.lambda_grid = lambda_grid
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.best_lambda_, self.cv_scores_ = _cv_select_lambda(
            self._base_estimator(), X, y, is_classifier=True, method=self.method,
            lambda_grid=self.lambda_grid, n_splits=self.n_splits,
            random_state=self.random_state)
        self.lam = self.best_lambda_
        return super().fit(X, y)