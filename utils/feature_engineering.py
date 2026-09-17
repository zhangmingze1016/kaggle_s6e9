import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from the original dataframe.
        """

        df = df.copy()

        # ============================================================
        # 1. Missing-value features
        # ============================================================

        # Number of missing values in each row
        df["missing_count"] = df.isna().sum(axis=1)

        # Whether the row contains any missing value
        df["has_missing"] = (df["missing_count"] > 0).astype(int)


        # ============================================================
        # 2. Numeric aggregate features
        # ============================================================

        numeric_cols = df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        # Avoid using ID as a numeric feature in aggregate statistics
        if "id" in numeric_cols:
            numeric_cols.remove("id")

        if numeric_cols:
            df["numeric_mean"] = df[numeric_cols].mean(axis=1)
            df["numeric_std"] = df[numeric_cols].std(axis=1)
            df["numeric_min"] = df[numeric_cols].min(axis=1)
            df["numeric_max"] = df[numeric_cols].max(axis=1)
            df["numeric_range"] = (
                df["numeric_max"] - df["numeric_min"]
            )


        # ============================================================
        # 3. Categorical count features
        # ============================================================

        categorical_cols = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if categorical_cols:
            # Number of distinct categorical values within one row
            df["categorical_unique_count"] = (
                df[categorical_cols]
                .astype(str)
                .nunique(axis=1)
            )


        # ============================================================
        # 4. Example interaction features
        # ============================================================

        # Add domain-specific interactions here.
        #
        # Example:
        #
        # if "Age" in df.columns and "Income" in df.columns:
        #     df["income_per_age"] = (
        #         df["Income"] / (df["Age"] + 1)
        #     )


        # ============================================================
        # 5. Infinite-value cleanup
        # ============================================================

        df = df.replace([np.inf, -np.inf], np.nan)

        return df