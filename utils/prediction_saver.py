from pathlib import Path
import pandas as pd


class PredictionSaver:

    def __init__(
        self,
        output_dir="predictions",
        experiment_file="experiments.csv"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_file = Path(experiment_file)

    def save(
        self,
        ids,
        predictions,
        score,
        params,
        fold_mean=None,
        fold_std=None,
        best_iterations=None,
        target="Will_Buy_EV"
    ):

        # ---------------------------------
        # Version
        # ---------------------------------

        existing = list(
            self.output_dir.glob("prediction_v*.csv")
        )

        version = len(existing) + 1
        version_name = f"v{version:03d}"

        # ---------------------------------
        # Model name
        # ---------------------------------

        model_name = params.get("model", "model")

        model_short = {
            "CatBoost": "cb",
            "XGBoost": "xgb",
            "LightGBM": "lgbm"
        }.get(model_name, model_name.lower())

        # ---------------------------------
        # Build filename
        # ---------------------------------

        filename_parts = [
            f"prediction_{version_name}",
            model_short
        ]

        if params.get("feature_engineering", False):
            filename_parts.append("fe")

        if params.get("feature_engineering") and params.get("feature_recipe") == "notebook":
            filename_parts.append("notebook")
        if params.get("target_encoding", False):
            filename_parts.append("te3")

        # CatBoost / common parameters
        if "depth" in params:
            filename_parts.append(
                f"d{params['depth']}"
            )

        # XGBoost uses max_depth
        elif "max_depth" in params:
            filename_parts.append(
                f"d{params['max_depth']}"
            )

        if "learning_rate" in params:
            filename_parts.append(
                f"lr{params['learning_rate']}"
            )

        if "iterations" in params:
            filename_parts.append(
                f"iter{params['iterations']}"
            )

        # XGBoost commonly uses n_estimators
        elif "n_estimators" in params:
            filename_parts.append(
                f"iter{params['n_estimators']}"
            )

        filename_parts.append(
            f"auc_{score:.5f}"
        )

        filename = (
            self.output_dir
            / ("_".join(filename_parts) + ".csv")
        )

        # ---------------------------------
        # Save Kaggle prediction
        # ---------------------------------

        predictions_df = pd.DataFrame({
            "id": ids,
            target: predictions
        })

        predictions_df.to_csv(
            filename,
            index=False
        )

        # ---------------------------------
        # Experiment information
        # ---------------------------------

        experiment = {
            "version": version_name,
            **params,
            "oof_auc": score,
            "fold_mean_auc": fold_mean,
            "fold_std_auc": fold_std,
        }

        if best_iterations is not None:

            experiment["best_iterations"] = ",".join(
                map(str, best_iterations)
            )

            experiment["mean_best_iteration"] = (
                sum(best_iterations)
                / len(best_iterations)
            )

        experiment_df = pd.DataFrame(
            [experiment]
        )

        # ---------------------------------
        # Append to experiments.csv
        # ---------------------------------

        if self.experiment_file.exists():

            previous = pd.read_csv(self.experiment_file)
            pd.concat([previous, experiment_df], ignore_index=True).to_csv(
                self.experiment_file,
                index=False,
            )

        else:

            experiment_df.to_csv(
                self.experiment_file,
                index=False
            )

        # ---------------------------------
        # Output
        # ---------------------------------

        print("\nExperiment Saved")
        print("==============================")
        print(f"Version:    {version_name}")
        print(f"Model:      {model_name}")
        print(f"Prediction: {filename}")
        print(f"OOF AUC:    {score:.5f}")

        print("\nParameters")
        print("==============================")

        for key, value in params.items():
            print(f"{key}: {value}")

        return filename


