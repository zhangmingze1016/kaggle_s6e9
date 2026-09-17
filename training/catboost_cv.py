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
        help="Enable feature engineering",
    )

    return parser.parse_args()


# ============================================================
# Main experiment
# ============================================================

def main():

    # ========================================================
    # 1. Read arguments
    # ========================================================

    args = parse_args()

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

    print("\n==============================")
    print("Experiment Parameters")
    print("==============================")

    for name, value in experiment_params.items():
        print(f"{name}: {value}")


    # ========================================================
    # 2. Load data
    # ========================================================

    loader = EVDataLoader(
        feature_engineering=args.feature_engineering
    )

    X, y, X_test, test_ids, cat_cols = loader.load()

    # Target:
    # Yes -> 1
    # No  -> 0

    y_binary = (
        y
        .eq("Yes")
        .astype(int)
    )


    # ========================================================
    # 3. Data information
    # ========================================================

    print("\n==============================")
    print("Data Information")
    print("==============================")

    print(f"Train shape: {X.shape}")
    print(f"Test shape:  {X_test.shape}")

    print(f"\nNumber of features: {X.shape[1]}")
    print(f"Categorical features: {len(cat_cols)}")

    if cat_cols:
        print("\nCategorical columns:")
        for column in cat_cols:
            print(f"  {column}")

    print("\nTarget distribution:")
    print(y_binary.value_counts(normalize=True))


    # ========================================================
    # 4. Cross-validation setup
    # ========================================================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # OOF predictions
    oof_pred = np.zeros(
        len(X),
        dtype=float,
    )

    # Test predictions averaged across folds
    test_pred = np.zeros(
        len(X_test),
        dtype=float,
    )

    fold_scores = []
    best_iterations = []


    # ========================================================
    # 5. Cross-validation
    # ========================================================

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y_binary),
        start=1,
    ):

        print("\n")
        print("=" * 60)
        print(f"Fold {fold}/{N_SPLITS}")
        print("=" * 60)


        # ----------------------------------------------------
        # Split fold
        # ----------------------------------------------------

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y_binary.iloc[train_idx]
        y_val = y_binary.iloc[val_idx]


        print(
            f"Train rows: {len(X_train)} | "
            f"Validation rows: {len(X_val)}"
        )


        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = CatBoostClassifier(

            # Number of boosting trees
            iterations=args.iterations,

            # Maximum tree depth
            depth=args.depth,

            # Step size
            learning_rate=args.learning_rate,

            # L2 regularization
            l2_leaf_reg=args.l2_leaf_reg,

            # Reproducibility
            random_seed=RANDOM_SEED,

            # Evaluation metric
            eval_metric="AUC",

            # Binary classification loss
            loss_function="Logloss",

            # Print training progress every 100 iterations
            verbose=100,

            # Automatically keep the best model
            use_best_model=True,

            # CPU training
            task_type="CPU",

            # Allow writing CatBoost temporary training files?
            allow_writing_files=False,
        )


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,

            cat_features=cat_cols,

            eval_set=(
                X_val,
                y_val,
            ),

            early_stopping_rounds=(
                args.early_stopping_rounds
            ),
        )


        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        val_pred = model.predict_proba(
            X_val
        )[:, 1]

        oof_pred[val_idx] = val_pred


        # ----------------------------------------------------
        # Validation metric
        # ----------------------------------------------------

        fold_auc = roc_auc_score(
            y_val,
            val_pred,
        )

        fold_scores.append(
            fold_auc
        )


        # ----------------------------------------------------
        # Best iteration
        # ----------------------------------------------------

        best_iteration = (
            model.get_best_iteration()
        )

        best_iterations.append(
            best_iteration
        )


        # ----------------------------------------------------
        # Fold results
        # ----------------------------------------------------

        print("\nFold Results")
        print("------------------------------")

        print(
            f"Fold {fold} AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )


        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        fold_test_pred = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_pred += (
            fold_test_pred / N_SPLITS
        )


    # ========================================================
    # 6. Final CV results
    # ========================================================

    mean_auc = np.mean(
        fold_scores
    )

    std_auc = np.std(
        fold_scores
    )

    oof_auc = roc_auc_score(
        y_binary,
        oof_pred,
    )


    print("\n")
    print("=" * 60)
    print("Final Cross-Validation Results")
    print("=" * 60)

    for fold, score in enumerate(
        fold_scores,
        start=1,
    ):

        print(
            f"Fold {fold}: "
            f"{score:.6f}"
        )


    print("\n------------------------------")

    print(
        f"Mean Fold AUC: "
        f"{mean_auc:.6f}"
    )

    print(
        f"Std Fold AUC:  "
        f"{std_auc:.6f}"
    )

    print(
        f"OOF ROC AUC:   "
        f"{oof_auc:.6f}"
    )


    # ========================================================
    # 7. Best iteration statistics
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Best Iterations")
    print("=" * 60)

    for fold, iteration in enumerate(
        best_iterations,
        start=1,
    ):

        print(
            f"Fold {fold}: "
            f"{iteration}"
        )


    mean_best_iteration = np.mean(
        best_iterations
    )

    std_best_iteration = np.std(
        best_iterations
    )

    print("\n------------------------------")

    print(
        f"Mean best iteration: "
        f"{mean_best_iteration:.0f}"
    )

    print(
        f"Std best iteration:  "
        f"{std_best_iteration:.1f}"
    )


    # ========================================================
    # 8. Prediction statistics
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Test Prediction Statistics")
    print("=" * 60)

    print(
        f"Mean prediction: "
        f"{test_pred.mean():.6f}"
    )

    print(
        f"Std prediction:  "
        f"{test_pred.std():.6f}"
    )

    print(
        f"Min prediction:  "
        f"{test_pred.min():.6f}"
    )

    print(
        f"Max prediction:  "
        f"{test_pred.max():.6f}"
    )


    # ========================================================
    # 9. Save predictions + experiment metadata
    # ========================================================

    PredictionSaver().save(
        ids=test_ids,
        predictions=test_pred,
        score=oof_auc,
        params=experiment_params,
        fold_mean=mean_auc,
        fold_std=std_auc,
        best_iterations=best_iterations,
    )


    print("\nExperiment completed successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()