import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

from hs_lib import (
    HSTreeClassifierCV, HSTreeRegressorCV,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 快速测试开关：第一次在新机器上跑，先设 QUICK_TEST=True，
# 用很小的设置（约10~20秒）确认整个流程没问题、环境装对了，
# 确认没问题后改成 False 再跑出完整结果（大约1~5分钟，取决于电脑性能）。
# ------------------------------------------------------------------
QUICK_TEST = False

if QUICK_TEST:
    N_SEEDS = 2
    MAX_LEAF_NODES_GRID = [8, 16]
    N_TREES_RF = 20
    LAMBDA_GRID = (1, 10, 50)
else:
    N_SEEDS = 10                       # 重复随机划分次数（论文里类似设置，用以估计方差）
    MAX_LEAF_NODES_GRID = [4, 8, 16, 32]   # 树复杂度网格（对应论文 Fig.2 的 x 轴：叶子数）
    N_TREES_RF = 50                    # 随机森林规模
    LAMBDA_GRID = (0.1, 1, 2, 5, 10, 25, 50, 100, 200)  # HS 收缩强度网格


# ----------------------------------------------------------------------
# 数据集配置
# ----------------------------------------------------------------------

def get_datasets():
    datasets = {}

    X, y = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer (binary, AUC)"] = dict(X=X, y=y, task="binary")

    X, y = load_wine(return_X_y=True)
    datasets["wine (3-class, accuracy)"] = dict(X=X, y=y, task="multiclass")

    X, y = load_diabetes(return_X_y=True)
    datasets["diabetes (regression, R2)"] = dict(X=X, y=y, task="regression")

    return datasets


def score(task, y_true, y_pred_or_proba):
    if task == "binary":
        return roc_auc_score(y_true, y_pred_or_proba)
    elif task == "multiclass":
        return accuracy_score(y_true, y_pred_or_proba)
    else:
        return r2_score(y_true, y_pred_or_proba)


# ----------------------------------------------------------------------
# 单次实验：给定数据集 + 树复杂度 + 随机种子，比较 4 个模型
#   CART        : 普通决策树（无收缩）
#   HS-CART     : 本文复现的 Hierarchical Shrinkage 应用到单棵树（CV选lambda）
#   RF          : 普通随机森林（无收缩）
#   HS-RF       : Hierarchical Shrinkage 应用到 RF 中每一棵树（CV选lambda）
# ----------------------------------------------------------------------

def run_single_trial(X, y, task, max_leaf_nodes, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed,
        stratify=y if task != "regression" else None,
    )

    results = {}

    if task == "regression":
        base_tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=seed)
        base_tree.fit(Xtr, ytr)
        results["CART"] = score(task, yte, base_tree.predict(Xte))

        hs_tree = HSTreeRegressorCV(
            estimator=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=seed),
            lambda_grid=LAMBDA_GRID, n_splits=3, random_state=seed,
        ).fit(Xtr, ytr)
        results["HS-CART"] = score(task, yte, hs_tree.predict(Xte))
        results["best_lambda_tree"] = hs_tree.best_lambda_

        base_rf = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, n_estimators=N_TREES_RF,
                                         random_state=seed)
        base_rf.fit(Xtr, ytr)
        results["RF"] = score(task, yte, base_rf.predict(Xte))

        hs_rf = HSTreeRegressorCV(
            estimator=RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, n_estimators=N_TREES_RF,
                                             random_state=seed),
            lambda_grid=LAMBDA_GRID, n_splits=3, random_state=seed,
        ).fit(Xtr, ytr)
        results["HS-RF"] = score(task, yte, hs_rf.predict(Xte))
        results["best_lambda_rf"] = hs_rf.best_lambda_

    else:
        base_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=seed)
        base_tree.fit(Xtr, ytr)
        proba = base_tree.predict_proba(Xte)
        results["CART"] = score(task, yte, proba[:, 1] if task == "binary" else np.argmax(proba, axis=1))

        hs_tree = HSTreeClassifierCV(
            estimator=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=seed),
            lambda_grid=LAMBDA_GRID, n_splits=3, random_state=seed,
        ).fit(Xtr, ytr)
        proba = hs_tree.predict_proba(Xte)
        results["HS-CART"] = score(task, yte, proba[:, 1] if task == "binary" else np.argmax(proba, axis=1))
        results["best_lambda_tree"] = hs_tree.best_lambda_

        base_rf = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, n_estimators=N_TREES_RF,
                                          random_state=seed)
        base_rf.fit(Xtr, ytr)
        proba = base_rf.predict_proba(Xte)
        results["RF"] = score(task, yte, proba[:, 1] if task == "binary" else np.argmax(proba, axis=1))

        hs_rf = HSTreeClassifierCV(
            estimator=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, n_estimators=N_TREES_RF,
                                              random_state=seed),
            lambda_grid=LAMBDA_GRID, n_splits=3, random_state=seed,
        ).fit(Xtr, ytr)
        proba = hs_rf.predict_proba(Xte)
        results["HS-RF"] = score(task, yte, proba[:, 1] if task == "binary" else np.argmax(proba, axis=1))
        results["best_lambda_rf"] = hs_rf.best_lambda_

    return results


def main():
    datasets = get_datasets()
    all_rows = []
    t0 = time.time()

    for dname, d in datasets.items():
        for mln in MAX_LEAF_NODES_GRID:
            for seed in range(N_SEEDS):
                t_iter = time.time()
                r = run_single_trial(d["X"], d["y"], d["task"], mln, seed)
                row = dict(dataset=dname, max_leaf_nodes=mln, seed=seed)
                row.update(r)
                all_rows.append(row)
                # 实时打印进度，每个(数据集,叶子数,随机种子)组合跑完就打一行，
                # 这样能亲眼看到程序在正常推进，不会误以为卡死
                print(f"  -> {dname} | max_leaf_nodes={mln} | seed={seed} "
                      f"完成，本次耗时 {time.time()-t_iter:.2f}s，累计 {time.time()-t0:.1f}s",
                      flush=True)
        print(f"[done] {dname}  (累计耗时 {time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv("results_raw.csv", index=False)

    # 汇总：每个数据集 x 每个叶子数设置，对4个模型求均值±标准差
    summary_rows = []
    for (dname, mln), g in df.groupby(["dataset", "max_leaf_nodes"]):
        row = {"dataset": dname, "max_leaf_nodes": mln}
        for col in ["CART", "HS-CART", "RF", "HS-RF"]:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std()
        row["mean_best_lambda_tree"] = g["best_lambda_tree"].mean()
        row["mean_best_lambda_rf"] = g["best_lambda_rf"].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results_summary.csv", index=False)

    print("\n==== 汇总结果（均值，跨 %d 个随机种子） ====" % N_SEEDS)
    pd.set_option("display.width", 160)
    print(summary[["dataset", "max_leaf_nodes",
                    "CART_mean", "HS-CART_mean",
                    "RF_mean", "HS-RF_mean"]].round(4))

    # 论文核心结论 (C1)(C2) 的整体验证：对每个数据集，HS对单树/对RF的
    # 平均提升幅度（跨所有叶子数设置和随机种子）
    print("\n==== 论文核心结论验证：HS 带来的平均性能提升 ====")
    for dname, g in df.groupby("dataset"):
        d_tree = (g["HS-CART"] - g["CART"]).mean()
        d_rf = (g["HS-RF"] - g["RF"]).mean()
        print(f"{dname:30s}  单树提升: {d_tree:+.4f}   RF提升: {d_rf:+.4f}")

    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()