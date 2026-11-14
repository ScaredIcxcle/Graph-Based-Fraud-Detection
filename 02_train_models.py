"""
02_train_models.py

Trains Model A (baseline features only) and Model B (baseline + graph
features including community_fraud_rate) with 5-fold CV, then evaluates
both on the held-out test set overall and segmented by pattern tags.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

import config

BASELINE_FEATURES = [
    "amt", "hour", "day_of_week", "age", "distance_km",
    "cust_txn_count_so_far", "cust_avg_amt_so_far", "minutes_since_prev_txn",
    "category",
]

GRAPH_FEATURES = [
    "cust_pagerank", "merch_pagerank",
    "cust_degree_weighted_frac", "merch_degree_weighted_frac",
    "cust_degree_frac", "merch_degree_frac",
    "cust_community_density", "community_fraud_rate",
    "merchant_indeg_frac", "customer_outdeg_frac",
]

TAG_COLS = ["tag_fan_in", "tag_fan_out", "tag_chain", "tag_dense_community"]


def load_features():
    print("Loading feature files...")
    train = pd.read_csv(f"{config.OUT_DIR}/train_features.csv")
    test = pd.read_csv(f"{config.OUT_DIR}/test_features.csv")
    print(f"  train: {len(train):,} rows, fraud rate {train['is_fraud'].mean():.4%}")
    print(f"  test:  {len(test):,} rows, fraud rate {test['is_fraud'].mean():.4%}")
    return train, test


def prep_features(df, feature_cols):
    X = df[feature_cols].copy()
    if "category" in X.columns:
        X = pd.get_dummies(X, columns=["category"], prefix="cat")
    X = X.fillna(0)
    return X


def align_columns(X_train, X_test):
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)
    return X_train, X_test


def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def run_cv(X, y, model_name):
    print(f"\nRunning 5-fold CV for {model_name}...")
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBClassifier(**config.XGB_PARAMS)
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]
        score = pr_auc_score(y_val, y_proba)
        fold_scores.append(score)
        print(f"  fold {fold+1}/{config.N_FOLDS}: PR-AUC = {score:.4f}")
    mean_score, std_score = np.mean(fold_scores), np.std(fold_scores)
    print(f"  {model_name} CV PR-AUC: {mean_score:.4f} +/- {std_score:.4f}")
    return fold_scores, mean_score, std_score


def fit_final_model(X, y, model_name):
    print(f"Fitting final {model_name} on full training set...")
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X, y)
    return model


def evaluate_on_test(model, X_test, y_test, model_name, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "n": len(y_test),
        "n_fraud": int(y_test.sum()),
        "pr_auc": pr_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }, y_proba


def segmented_evaluation(test_df, y_test, probaA, probaB, threshold=0.5):
    results = []
    for tag in TAG_COLS:
        mask = test_df[tag] == 1
        n_subset = mask.sum()
        if n_subset < 20 or y_test[mask].sum() < 5:
            print(f"  skipping segment {tag}: too few rows/fraud cases")
            continue
        yA_sub, yB_sub = probaA[mask], probaB[mask]
        y_sub = y_test[mask]
        for name, proba in [("Model A (baseline)", yA_sub), ("Model B (graph-enhanced)", yB_sub)]:
            pred = (proba >= threshold).astype(int)
            results.append({
                "segment": tag,
                "model": name,
                "n": int(n_subset),
                "n_fraud": int(y_sub.sum()),
                "pr_auc": pr_auc_score(y_sub, proba),
                "f1": f1_score(y_sub, pred, zero_division=0),
                "precision": precision_score(y_sub, pred, zero_division=0),
                "recall": recall_score(y_sub, pred, zero_division=0),
            })
    return pd.DataFrame(results)


def main():
    train, test = load_features()
    y_train = train["is_fraud"]
    y_test = test["is_fraud"]

    XA_train = prep_features(train, BASELINE_FEATURES)
    XA_test = prep_features(test, BASELINE_FEATURES)
    XA_train, XA_test = align_columns(XA_train, XA_test)

    XB_train = prep_features(train, BASELINE_FEATURES + GRAPH_FEATURES)
    XB_test = prep_features(test, BASELINE_FEATURES + GRAPH_FEATURES)
    XB_train, XB_test = align_columns(XB_train, XB_test)

    cv_A_scores, cv_A_mean, cv_A_std = run_cv(XA_train, y_train, "Model A (baseline)")
    cv_B_scores, cv_B_mean, cv_B_std = run_cv(XB_train, y_train, "Model B (graph-enhanced)")

    cv_results = pd.DataFrame({
        "fold": list(range(1, config.N_FOLDS + 1)) * 2,
        "model": ["Model A (baseline)"] * config.N_FOLDS + ["Model B (graph-enhanced)"] * config.N_FOLDS,
        "pr_auc": cv_A_scores + cv_B_scores,
    })
    cv_results.to_csv(f"{config.OUT_DIR}/cv_results.csv", index=False)

    modelA = fit_final_model(XA_train, y_train, "Model A (baseline)")
    modelB = fit_final_model(XB_train, y_train, "Model B (graph-enhanced)")

    resultA, probaA = evaluate_on_test(modelA, XA_test, y_test, "Model A (baseline)")
    resultB, probaB = evaluate_on_test(modelB, XB_test, y_test, "Model B (graph-enhanced)")

    print("\n=== OVERALL TEST SET RESULTS ===")
    overall_df = pd.DataFrame([resultA, resultB])
    print(overall_df.to_string(index=False))

    print("\n=== SEGMENTED EVALUATION (by pattern tag) ===")
    segmented_df = segmented_evaluation(test, y_test, probaA, probaB)
    print(segmented_df.to_string(index=False))

    importances = pd.DataFrame({
        "feature": XB_train.columns,
        "importance": modelB.feature_importances_,
    }).sort_values("importance", ascending=False)
    graph_importances = importances[importances["feature"].isin(GRAPH_FEATURES)]

    with open(f"{config.OUT_DIR}/model_comparison_report.txt", "w") as f:
        f.write("CROSS-VALIDATION RESULTS (5-fold, train set)\n")
        f.write(f"Model A (baseline) PR-AUC:        {cv_A_mean:.4f} +/- {cv_A_std:.4f}\n")
        f.write(f"Model B (graph-enhanced) PR-AUC:  {cv_B_mean:.4f} +/- {cv_B_std:.4f}\n\n")
        f.write("OVERALL TEST SET RESULTS\n")
        f.write(overall_df.to_string(index=False) + "\n\n")
        f.write("SEGMENTED EVALUATION (by pattern tag, test set)\n")
        f.write(segmented_df.to_string(index=False) + "\n\n")
        f.write("TOP GRAPH FEATURE IMPORTANCES (Model B)\n")
        f.write(graph_importances.to_string(index=False) + "\n\n")
        f.write("ALL FEATURE IMPORTANCES (Model B)\n")
        f.write(importances.to_string(index=False) + "\n")

    print(f"\nFull report saved to {config.OUT_DIR}/model_comparison_report.txt")


if __name__ == "__main__":
    main()