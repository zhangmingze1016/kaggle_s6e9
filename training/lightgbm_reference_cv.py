import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils.data_loader import EVDataLoader
from utils.feature_engineering import TripleTargetEncoder
from utils.prediction_saver import PredictionSaver


# ============================================================
# Default configuration
# ============================================================

N_SPLITS = 10
RANDOM_SEED = 42


DEFAULT_PARAMS = {
    "n_estimators": 100000,
    "learning_rate": 0.005,
    "num_leaves": 31,
    "max_depth": 7,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "early_stopping_rounds": 500,
}


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="LightGBM CV with notebook FE-B, digits, frequency and triple TE"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_PARAMS["n_estimators"],
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_PARAMS["learning_rate"],
    )

    parser.add_argument(
        "--num-leaves",
        type=int,
        default=DEFAULT_PARAMS["num_leaves"],
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_PARAMS["max_depth"],
    )

    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=DEFAULT_PARAMS["min_child_samples"],
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=DEFAULT_PARAMS["subsample"],
    )

    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=DEFAULT_PARAMS["colsample_bytree"],
    )

    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=DEFAULT_PARAMS["reg_lambda"],
    )

    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=DEFAULT_PARAMS["reg_alpha"],
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_PARAMS["early_stopping_rounds"],
    )

    parser.add_argument(
        "--feature-engineering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable feature engineering (default: enabled)",
    )

    parser.add_argument("--feature-recipe", choices=["notebook", "legacy"], default="notebook")
    parser.add_argument("--target-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--te-cv", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--is-unbalance", action=argparse.BooleanOptionalAction, default=True)
    # Notebook leaves subsample_freq at 0: subsample=0.8 is inactive there.
    parser.add_argument("--subsample-freq", type=int, default=0)
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--experiment-file", default="experiments.csv")
    return parser.parse_args()


# ============================================================
# Main training pipeline
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # 1. Experiment parameters
    # ========================================================

    experiment_params = {

        "model": "LightGBM",

        "feature_engineering": (
            args.feature_engineering
        ),

        "n_estimators": (
            args.n_estimators
        ),

        "learning_rate": (
            args.learning_rate
        ),

        "num_leaves": (
            args.num_leaves
        ),

        "max_depth": (
            args.max_depth
        ),

        "min_child_samples": (
            args.min_child_samples
        ),

        "subsample": (
            args.subsample
        ),

        "colsample_bytree": (
            args.colsample_bytree
        ),

        "reg_lambda": (
            args.reg_lambda
        ),

        "reg_alpha": (
            args.reg_alpha
        ),

        "early_stopping_rounds": (
            args.early_stopping_rounds
        ),

        "n_splits": args.n_splits,
        "feature_recipe": args.feature_recipe if args.feature_engineering else "raw",
        "target_encoding": args.target_encoding,
        "te_cv": args.te_cv,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "max_bin": args.max_bin,
        "is_unbalance": args.is_unbalance,
        "subsample_freq": args.subsample_freq,
        "train_path": args.train_path,
        "test_path": args.test_path,

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
        feature_engineering=args.feature_engineering,
        feature_recipe=args.feature_recipe,
        train_path=args.train_path,
        test_path=args.test_path,
    )

    X, y, X_test, test_ids, cat_cols = loader.load()


    # --------------------------------------------------------
    # Binary target
    # --------------------------------------------------------

    y_binary = (
        y.eq("Yes").astype(int)
    )


    print("\n==============================")
    print("Data Information")
    print("==============================")

    print(
        f"Train shape: {X.shape}"
    )

    print(
        f"Test shape:  {X_test.shape}"
    )

    print(
        f"Categorical features: "
        f"{len(cat_cols)}"
    )


    # ========================================================
    # 3. Prepare categorical features
    # ========================================================
    #
    # LightGBM can directly use pandas categorical columns.
    #
    # We combine train + test temporarily so their category
    # mappings are exactly the same.
    #
    # ========================================================

    n_train = len(X)

    combined = pd.concat(
        [
            X,
            X_test,
        ],
        axis=0,
        ignore_index=True,
    )


    for col in cat_cols:

        combined[col] = (
            combined[col]
            .astype("category")
        )


    X = (
        combined
        .iloc[:n_train]
        .copy()
    )

    X_test = (
        combined
        .iloc[n_train:]
        .copy()
    )

    X_test.reset_index(
        drop=True,
        inplace=True,
    )


    # ========================================================
    # 4. Cross-validation
    # ========================================================

    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


    # --------------------------------------------------------
    # Prediction containers
    # --------------------------------------------------------

    oof_pred = np.zeros(
        len(X),
        dtype=float,
    )

    test_pred = np.zeros(
        len(X_test),
        dtype=float,
    )

    fold_scores = []

    best_iterations = []


    # ========================================================
    # 5. Stratified fold training
    # ========================================================

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            X,
            y_binary,
        ),
        start=1,
    ):

        print("\n")
        print("=" * 60)

        print(
            f"Fold {fold}/{args.n_splits}"
        )

        print("=" * 60)


        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        X_train = (
            X.iloc[train_idx]
        )

        X_val = (
            X.iloc[val_idx]
        )


        y_train = (
            y_binary.iloc[train_idx]
        )

        y_val = (
            y_binary.iloc[val_idx]
        )


        print(
            f"Train rows: {len(X_train)} | "
            f"Validation rows: {len(X_val)}"
        )


        # Fit encodings only on this outer training fold. Training rows use
        # five-way inner cross-fitting; validation/test use its fitted maps.
        X_fold_test = X_test
        if args.target_encoding:
            encoder = TripleTargetEncoder(cv=args.te_cv, random_state=RANDOM_SEED)
            X_train = encoder.fit_transform(X_train, y_train)
            X_val = encoder.transform(X_val)
            X_fold_test = encoder.transform(X_test)

        # ====================================================
        # Model
        # ====================================================

        model = LGBMClassifier(

            objective="binary",

            n_estimators=(
                args.n_estimators
            ),

            learning_rate=(
                args.learning_rate
            ),

            num_leaves=(
                args.num_leaves
            ),

            max_depth=(
                args.max_depth
            ),

            min_child_samples=(
                args.min_child_samples
            ),

            subsample=(
                args.subsample
            ),

            colsample_bytree=(
                args.colsample_bytree
            ),

            reg_lambda=(
                args.reg_lambda
            ),

            reg_alpha=(
                args.reg_alpha
            ),

            random_state=(
                RANDOM_SEED
            ),

            n_jobs=args.n_jobs,
            device=args.device,
            max_bin=args.max_bin,
            is_unbalance=args.is_unbalance,
            subsample_freq=args.subsample_freq,
            metric="auc",

            verbosity=-1,
        )


        # ====================================================
        # Train
        # ====================================================

        model.fit(

            X_train,

            y_train,

            eval_X=X_val,
            eval_y=y_val,

            eval_metric="auc",

            categorical_feature=cat_cols,

            callbacks=[

                lgb.early_stopping(
                    stopping_rounds=(
                        args.early_stopping_rounds
                    ),
                    first_metric_only=True,
                    verbose=True,
                ),

                lgb.log_evaluation(
                    period=100
                ),
            ],
        )


        # ====================================================
        # Validation predictions
        # ====================================================

        val_pred = (
            model.predict_proba(
                X_val,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )


        oof_pred[val_idx] = (
            val_pred
        )


        # ====================================================
        # Fold score
        # ====================================================

        fold_auc = (
            roc_auc_score(
                y_val,
                val_pred,
            )
        )


        fold_scores.append(
            fold_auc
        )


        # ====================================================
        # Best iteration
        # ====================================================

        best_iteration = (
            model.best_iteration_
        )

        best_iterations.append(
            best_iteration
        )


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


        # ====================================================
        # Test predictions
        # ====================================================

        fold_test_pred = (
            model.predict_proba(
                X_fold_test,
                num_iteration=(
                    model.best_iteration_
                ),
            )[:, 1]
        )


        test_pred += (
            fold_test_pred
            / args.n_splits
        )


    # ========================================================
    # 6. Final CV results
    # ========================================================

    mean_auc = (
        np.mean(
            fold_scores
        )
    )


    std_auc = (
        np.std(
            fold_scores
        )
    )


    oof_auc = (
        roc_auc_score(
            y_binary,
            oof_pred,
        )
    )


    print("\n")
    print("=" * 60)

    print(
        "Final Cross-Validation Results"
    )

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
    # 7. Best iterations
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "Best Iterations"
    )

    print("=" * 60)


    for fold, iteration in enumerate(
        best_iterations,
        start=1,
    ):

        print(
            f"Fold {fold}: "
            f"{iteration}"
        )


    mean_best_iteration = (
        np.mean(
            best_iterations
        )
    )

    std_best_iteration = (
        np.std(
            best_iterations
        )
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
    # 8. Test prediction statistics
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "Test Prediction Statistics"
    )

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
    # 9. Save experiment
    # ========================================================

    PredictionSaver(
        output_dir=args.output_dir, experiment_file=args.experiment_file,
    ).save(

        ids=test_ids,

        predictions=test_pred,

        score=oof_auc,

        params=experiment_params,

        fold_mean=mean_auc,

        fold_std=std_auc,

        best_iterations=best_iterations,
    )


    print(
        "\nExperiment completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
