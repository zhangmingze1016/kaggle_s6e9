import numpy as np
import pandas as pd


class FeatureEngineer:

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # ====================================================
        # Missing information
        # ====================================================

        df["missing_count"] = (
            df.isna().sum(axis=1)
        )

        # ====================================================
        # EV-specific interaction features
        # ====================================================

        # 在这里添加真正的特征
        #
        # Example:
        #
        # if (
        #     "Annual_Income" in df.columns
        #     and "Vehicle_Price" in df.columns
        # ):
        #
        #     df["price_income_ratio"] = (
        #         df["Vehicle_Price"]
        #         / (df["Annual_Income"] + 1)
        #     )

        # ====================================================
        # Clean infinite values
        # ====================================================

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return df