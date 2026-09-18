import pandas as pd

from utils.feature_engineering import FeatureEngineer, NotebookFeatureEngineer, MultiScaleFeatureEngineer


class EVDataLoader:

    def __init__(
        self,
        train_path="data/train.csv",
        test_path="data/test.csv",
        feature_engineering=False,
        feature_recipe="legacy",
        sample_size=None,
        random_state=42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_engineering = feature_engineering
        if feature_recipe not in {"legacy", "notebook", "multiscale"}:
            raise ValueError("Unknown feature_recipe")
        self.feature_recipe = feature_recipe
        self.sample_size = sample_size
        self.random_state = random_state

    def load(self):

        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        if not train['id'].is_unique or not test['id'].is_unique:
            raise ValueError('IDs must be unique')
        if not train['Will_Buy_EV'].isin(['No', 'Yes']).all():
            raise ValueError('Target must contain only No/Yes')
        if self.sample_size is not None and self.sample_size < len(train):
            from sklearn.model_selection import train_test_split
            train, _ = train_test_split(
                train, train_size=self.sample_size, stratify=train['Will_Buy_EV'],
                random_state=self.random_state,
            )
            train = train.sort_index().reset_index(drop=True)
        self.train_ids = train['id'].copy()

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
            elif self.feature_recipe == "multiscale":
                X = MultiScaleFeatureEngineer.transform(X)
                X_test = MultiScaleFeatureEngineer.transform(X_test)
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
