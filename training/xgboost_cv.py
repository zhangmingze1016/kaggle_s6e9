import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from utils.data_loader import EVDataLoader
from utils.prediction_saver import PredictionSaver


# ============================================================
# Default configuration
# ============================================================

N_SPLITS = 5
RANDOM_SEED = 42


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="XGBoost 5-Fold Cross Validation"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=6
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0
    )

    return parser.parse_args()


# ============================================================
# Main training pipeline
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    loader = EVDataLoader()

    X, y, X_test, test_ids, cat_cols = loader.load()
    y = (y == "Yes").astype(int)

    print("\nData Loaded")
    print("==============================")
    print(f"Train shape: {X.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Categorical features: {cat_cols}")


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------
    #
    # CatBoost can directly handle string categorical columns.
    # XGBoost needs numerical input in this pipeline.
    #
    # Concatenate train + test before one-hot encoding so both
    # datasets receive exactly the same dummy columns.
    # --------------------------------------------------------

    n_train = len(X)

    combined = pd.concat(
        [X, X_test],
        axis=0,
        ignore_index=True
    )

    combined = pd.get_dummies(
        combined,
        columns=cat_cols,
        dtype=np.int8
    )

    X = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    X_test.reset_index(
        drop=True,
        inplace=True
    )

    print("\nAfter One-Hot Encoding")
    print("==============================")
    print(f"Train shape: {X.shape}")
    print(f"Test shape:  {X_test.shape}")


    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )


    # --------------------------------------------------------
    # Prediction containers
    # --------------------------------------------------------

    oof_pred = np.zeros(
        len(X),
        dtype=float
    )

    test_pred = np.zeros(
        len(X_test),
        dtype=float
    )

    fold_scores = []
    best_iterations = []


    # --------------------------------------------------------
    # 5-Fold CV
    # --------------------------------------------------------

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(
            f"\n========== Fold {fold} =========="
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,

            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,

            min_child_weight=args.min_child_weight,

            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,

            objective="binary:logistic",
            eval_metric="auc",

            random_state=RANDOM_SEED,

            tree_method="hist",

            n_jobs=-1,

            early_stopping_rounds=args.early_stopping_rounds
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_valid, y_valid)
            ],
            verbose=100
        )


        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        valid_pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof_pred[valid_idx] = valid_pred


        # ----------------------------------------------------
        # Fold AUC
        # ----------------------------------------------------

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        fold_scores.append(
            fold_auc
        )


        # ----------------------------------------------------
        # Best iteration
        # ----------------------------------------------------

        best_iteration = model.best_iteration

        best_iterations.append(
            best_iteration
        )

        print(
            f"Fold {fold} AUC: "
            f"{fold_auc:.5f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )


        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        fold_test_pred = model.predict_proba(
            X_test
        )[:, 1]

        test_pred += (
            fold_test_pred / N_SPLITS
        )


    # ========================================================
    # CV Results
    # ========================================================

    mean_auc = np.mean(
        fold_scores
    )

    std_auc = np.std(
        fold_scores
    )

    oof_auc = roc_auc_score(
        y,
        oof_pred
    )


    print("\n==============================")
    print("CV Results")
    print("==============================")

    for fold, score in enumerate(
        fold_scores,
        start=1
    ):
        print(
            f"Fold {fold}: "
            f"{score:.5f}"
        )

    print(
        f"\nMean Fold AUC: "
        f"{mean_auc:.5f}"
    )

    print(
        f"Std Fold AUC:  "
        f"{std_auc:.5f}"
    )

    print(
        f"OOF ROC AUC:   "
        f"{oof_auc:.5f}"
    )


    # ========================================================
    # Best iterations
    # ========================================================

    print("\nBest Iterations")
    print("==============================")

    for fold, iteration in enumerate(
        best_iterations,
        start=1
    ):
        print(
            f"Fold {fold}: "
            f"{iteration}"
        )

    mean_best_iteration = int(
        np.mean(best_iterations)
    )

    print(
        f"Mean best iteration: "
        f"{mean_best_iteration}"
    )


    # ========================================================
    # Experiment parameters
    # ========================================================

    experiment_params = {

        "model": "XGBoost",

        "max_depth": args.max_depth,

        "learning_rate": (
            args.learning_rate
        ),

        "n_estimators": (
            args.n_estimators
        ),

        "early_stopping_rounds": (
            args.early_stopping_rounds
        ),

        "subsample": (
            args.subsample
        ),

        "colsample_bytree": (
            args.colsample_bytree
        ),

        "min_child_weight": (
            args.min_child_weight
        ),

        "reg_lambda": (
            args.reg_lambda
        ),

        "reg_alpha": (
            args.reg_alpha
        ),

        "n_splits": N_SPLITS,

        "random_seed": RANDOM_SEED
    }


    # ========================================================
    # Save prediction
    # ========================================================

    PredictionSaver().save(

        ids=test_ids,

        predictions=test_pred,

        score=oof_auc,

        params=experiment_params,

        fold_mean=mean_auc,

        fold_std=std_auc,

        best_iterations=best_iterations
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()