import pandas as pd

from utils.feature_engineering import FeatureEngineer, NotebookFeatureEngineer


class EVDataLoader:

    def __init__(
        self,
        train_path="data/train.csv",
        test_path="data/test.csv",
        feature_engineering=False,
        feature_recipe="legacy",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_engineering = feature_engineering
        if feature_recipe not in {"legacy", "notebook"}:
            raise ValueError("feature_recipe must be legacy or notebook")
        self.feature_recipe = feature_recipe

    def load(self):

        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        # ---------------------------------------------
        # Target / ID
        # ---------------------------------------------

        y = train["Will_Buy_EV"]

        test_ids = test["id"].copy()

        X = train.drop(
            columns=["Will_Buy_EV", "id"]
        )

        X_test = test.drop(
            columns=["id"]
        )

        # ---------------------------------------------
        # Feature engineering
        # ---------------------------------------------

        if self.feature_engineering:

            if self.feature_recipe == "notebook":
                X, X_test = NotebookFeatureEngineer().transform_pair(X, X_test)
            else:
                engineer = FeatureEngineer()
                X = engineer.transform(X)
                X_test = engineer.transform(X_test)

        # ---------------------------------------------
        # Detect categorical columns
        # ---------------------------------------------

        if not X.columns.equals(X_test.columns):
            raise ValueError("Train and test features do not match")

        cat_cols = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        return (
            X,
            y,
            X_test,
            test_ids,
            cat_cols,
        )
