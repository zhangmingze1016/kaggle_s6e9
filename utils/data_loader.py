import pandas as pd

from utils.feature_engineering import FeatureEngineer


class EVDataLoader:

    def __init__(
        self,
        train_path="data/train.csv",
        test_path="data/test.csv",
        feature_engineering=False,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_engineering = feature_engineering

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

            engineer = FeatureEngineer()

            X = engineer.transform(X)
            X_test = engineer.transform(X_test)

        # ---------------------------------------------
        # Detect categorical columns
        # ---------------------------------------------

        cat_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        return (
            X,
            y,
            X_test,
            test_ids,
            cat_cols,
        )