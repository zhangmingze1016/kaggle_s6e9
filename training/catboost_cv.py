import argparse
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils.data_loader import EVDataLoader
from utils.prediction_saver import PredictionSaver


# ============================================================
# Default experiment configuration
# ============================================================

DEFAULT_PARAMS = {
    "iterations": 3000,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "early_stopping_rounds": 200,
}

N_SPLITS = 5
RANDOM_SEED = 42


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CatBoost cross-validation experiment"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_PARAMS["iterations"],
        help="Maximum number of boosting iterations",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_PARAMS["depth"],
        help="Tree depth",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_PARAMS["learning_rate"],
        help="Learning rate",
    )

    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=DEFAULT_PARAMS["l2_leaf_reg"],
        help="L2 regularization coefficient",
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_PARAMS["early_stopping_rounds"],
        help="Early stopping patience",
    )

    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        help="Add deterministic EV interaction features",
    )

    return parser.parse_args()


# ============================================================
# Main experiment
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Actual parameters used in this experiment
    # --------------------------------------------------------

    experiment_params = {
        "model": "CatBoost",
        "feature_engineering": args.feature_engineering,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
    }

    print("\nExperiment Parameters")
    print("==============================")

    for name, value in experiment_params.items():
        print(f"{name}: {value}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    loader = EVDataLoader(feature_engineering=args.feature_engineering)

    X, y, X_test, test_ids, cat_cols = loader.load()

    # Convert target to 0 / 1 for ROC AUC
    y_binary = (y == "Yes").astype(int)

    # --------------------------------------------------------
    # Cross-validation setup
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # Out-of-fold validation predictions
    oof_pred = np.zeros(len(X))

    # Averaged test predictions across all folds
    test_pred = np.zeros(len(X_test))

    fold_scores = []
    best_iterations = []

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y_binary),
        start=1,
    ):

        print(f"\n========== Fold {fold} ==========")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y_binary.iloc[train_idx]
        y_val = y_binary.iloc[val_idx]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=RANDOM_SEED,
            eval_metric="AUC",
            verbose=100,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            early_stopping_rounds=args.early_stopping_rounds,
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        val_pred = model.predict_proba(X_val)[:, 1]

        oof_pred[val_idx] = val_pred

        fold_auc = roc_auc_score(
            y_val,
            val_pred,
        )

        fold_scores.append(fold_auc)

        # ----------------------------------------------------
        # Best iteration
        # ----------------------------------------------------

        best_iteration = model.get_best_iteration()

        best_iterations.append(best_iteration)

        print(f"Fold {fold} AUC: {fold_auc:.5f}")
        print(f"Best iteration: {best_iteration}")

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        fold_test_pred = model.predict_proba(X_test)[:, 1]

        test_pred += fold_test_pred / N_SPLITS

    # --------------------------------------------------------
    # Final CV metrics
    # --------------------------------------------------------

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)

    oof_auc = roc_auc_score(
        y_binary,
        oof_pred,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n==============================")
    print("CV Results")
    print("==============================")

    for fold, score in enumerate(
        fold_scores,
        start=1,
    ):
        print(f"Fold {fold}: {score:.5f}")

    print()
    print(f"Mean Fold AUC: {mean_auc:.5f}")
    print(f"Std Fold AUC:  {std_auc:.5f}")
    print(f"OOF ROC AUC:   {oof_auc:.5f}")

    # --------------------------------------------------------
    # Best iterations
    # --------------------------------------------------------

    print("\nBest Iterations")
    print("==============================")

    for fold, iteration in enumerate(
        best_iterations,
        start=1,
    ):
        print(f"Fold {fold}: {iteration}")

    mean_best_iteration = np.mean(best_iterations)

    print(
        f"Mean best iteration: "
        f"{mean_best_iteration:.0f}"
    )

    # --------------------------------------------------------
    # Save prediction + experiment information
    # --------------------------------------------------------

    PredictionSaver().save(
        ids=test_ids,
        predictions=test_pred,
        score=oof_auc,
        params=experiment_params,
        fold_mean=mean_auc,
        fold_std=std_auc,
        best_iterations=best_iterations,
    )


if __name__ == "__main__":
    main()
