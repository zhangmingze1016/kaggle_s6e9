import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils.data_loader import EVDataLoader
from utils.prediction_saver import PredictionSaver


# =========================
# Config
# =========================

N_SPLITS = 5
RANDOM_SEED = 42

MODEL_PARAMS = {
    "iterations": 3000,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "eval_metric": "AUC",
    "verbose": 100,
}


# =========================
# Data
# =========================

loader = EVDataLoader()

X, y, X_test, test_ids, cat_cols = loader.load()

y_binary = (y == "Yes").astype(int)


# =========================
# Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

# 每一行训练数据最终都会得到一个 OOF prediction
oof_pred = np.zeros(len(X))

# 每个 fold 都预测一次 test
test_pred = np.zeros(len(X_test))

fold_scores = []


# =========================
# Train each fold
# =========================

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):
    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = CatBoostClassifier(**MODEL_PARAMS)

    model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    )

    # Validation prediction
    val_prob = model.predict_proba(X_val)[:, 1]

    # 保存到对应的原始位置
    oof_pred[val_idx] = val_prob

    # Fold AUC
    fold_auc = roc_auc_score(
        (y_val == "Yes").astype(int),
        val_prob,
    )

    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")
    print(f"Best iteration: {model.get_best_iteration()}")

    print(f"Fold {fold} AUC: {fold_auc:.5f}")

    # Test prediction
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS


# =========================
# Final CV score
# =========================

oof_auc = roc_auc_score(
    y_binary,
    oof_pred,
)

print("\n==============================")
print("CV Results")
print("==============================")

for fold, score in enumerate(fold_scores, start=1):
    print(f"Fold {fold}: {score:.5f}")

print(f"\nMean Fold AUC: {np.mean(fold_scores):.5f}")
print(f"Std Fold AUC:  {np.std(fold_scores):.5f}")
print(f"OOF ROC AUC:   {oof_auc:.5f}")


# =========================
# Save test prediction
# =========================

PredictionSaver().save(
    ids=test_ids,
    predictions=test_pred,
    score=oof_auc,
)