import numpy as np
import pandas as pd


class FeatureEngineer:

    @staticmethod
    def _combine_categories(
        df: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> pd.Series:
        """
        Combine two categorical columns into one interaction feature.
        """

        left = (
            df[col1]
            .astype("string")
            .fillna("__MISSING__")
        )

        right = (
            df[col2]
            .astype("string")
            .fillna("__MISSING__")
        )

        return left + "__" + right


    @staticmethod
    def _yes_no_to_int(
        series: pd.Series,
    ) -> pd.Series:
        """
        Convert Yes/No categorical values into 1/0.
        Unknown values become NaN.
        """

        return (
            series
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
            })
        )


    @staticmethod
    def _range_anxiety_to_int(
        series: pd.Series,
    ) -> pd.Series:
        """
        Convert ordinal range anxiety:
        Low -> 0
        Medium -> 1
        High -> 2
        """

        return (
            series
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "low": 0,
                "medium": 1,
                "high": 2,
            })
        )


    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # ====================================================
        # 1. Missing information
        # ====================================================

        df["missing_count"] = (
            df.isna().sum(axis=1)
        )


        # ====================================================
        # 2. Charging infrastructure
        # ====================================================

        home_stations = (
            df["Charging_Stations_Near_Home"]
        )

        work_stations = (
            df["Charging_Stations_Near_Work"]
        )

        # Total charging infrastructure available nearby
        df["Total_Charging_Stations"] = (
            home_stations
            + work_stations
        )

        # Difference between home/work charging availability
        df["Charging_Station_Difference"] = (
            home_stations
            - work_stations
        )

        # Absolute imbalance
        df["Charging_Station_Imbalance"] = (
            (
                home_stations
                - work_stations
            )
            .abs()
        )

        # Best charging availability at either location
        df["Max_Charging_Stations"] = (
            pd.concat(
                [
                    home_stations,
                    work_stations,
                ],
                axis=1,
            )
            .max(axis=1)
        )

        # Charging availability at weakest location
        df["Min_Charging_Stations"] = (
            pd.concat(
                [
                    home_stations,
                    work_stations,
                ],
                axis=1,
            )
            .min(axis=1)
        )

        # Whether there is any nearby public charging
        df["Has_Nearby_Charging"] = (
            df["Total_Charging_Stations"] > 0
        ).astype(int)


        # ====================================================
        # 3. Commute × charging interactions
        # ====================================================

        commute = df["Daily_Commute_km"]

        # Longer commute with fewer chargers should represent
        # a different EV-use situation from short commute.
        df["Commute_Per_Charging_Station"] = (
            commute
            / (
                df["Total_Charging_Stations"]
                + 1
            )
        )

        df["Charging_Stations_Per_Commute"] = (
            df["Total_Charging_Stations"]
            / (
                commute
                + 1
            )
        )


        # ====================================================
        # 4. Income / affordability features
        # ====================================================

        income = df["Annual_Income_USD"]

        cars = df["Number_of_Cars_Owned"]

        # Log income reduces extreme scale
        df["Log_Annual_Income"] = (
            np.log1p(
                income.clip(lower=0)
            )
        )

        # Income relative to household vehicle ownership
        df["Income_Per_Car_Plus_One"] = (
            income
            / (
                cars
                + 1
            )
        )

        # Income relative to daily travel requirement
        df["Income_Per_Commute_km"] = (
            income
            / (
                commute
                + 1
            )
        )


        # ====================================================
        # 5. Commute transformation
        # ====================================================

        df["Log_Daily_Commute"] = (
            np.log1p(
                commute.clip(lower=0)
            )
        )


        # ====================================================
        # 6. Binary categorical representations
        # ====================================================

        home_charging_flag = (
            self._yes_no_to_int(
                df["Home_Charging_Possible"]
            )
        )

        subsidy_flag = (
            self._yes_no_to_int(
                df["Subsidy_Available"]
            )
        )

        df["Home_Charging_Flag"] = (
            home_charging_flag
        )

        df["Subsidy_Flag"] = (
            subsidy_flag
        )


        # ====================================================
        # 7. Range anxiety as ordinal information
        # ====================================================

        range_anxiety = (
            self._range_anxiety_to_int(
                df["Range_Anxiety_Level"]
            )
        )

        df["Range_Anxiety_Ordinal"] = (
            range_anxiety
        )


        # ====================================================
        # 8. Behavioral interactions
        # ====================================================

        # Long commute may matter more for users with
        # stronger range anxiety.
        df["Commute_x_Range_Anxiety"] = (
            commute
            * range_anxiety
        )

        # Public charging availability interacts with
        # range anxiety.
        df["Charging_x_Range_Anxiety"] = (
            df["Total_Charging_Stations"]
            * range_anxiety
        )

        # Environmental concern may interact with income.
        df["Income_x_Environmental_Concern"] = (
            df["Log_Annual_Income"]
            * df["Environmental_Concern_Level"]
        )

        # Environmental concern may have different effects
        # depending on subsidy availability.
        df["Environmental_Concern_x_Subsidy"] = (
            df["Environmental_Concern_Level"]
            * subsidy_flag
        )

        # Long commute may be especially relevant when
        # home charging is unavailable.
        df["Commute_x_No_Home_Charging"] = (
            commute
            * (
                1 - home_charging_flag
            )
        )


        # ====================================================
        # 9. Categorical interaction features
        # ====================================================

        df["City_x_HomeCharging"] = (
            self._combine_categories(
                df,
                "City_Type",
                "Home_Charging_Possible",
            )
        )

        df["CarType_x_HomeCharging"] = (
            self._combine_categories(
                df,
                "Current_Car_Type",
                "Home_Charging_Possible",
            )
        )

        df["Anxiety_x_HomeCharging"] = (
            self._combine_categories(
                df,
                "Range_Anxiety_Level",
                "Home_Charging_Possible",
            )
        )

        df["Subsidy_x_CarType"] = (
            self._combine_categories(
                df,
                "Subsidy_Available",
                "Current_Car_Type",
            )
        )

        df["City_x_RangeAnxiety"] = (
            self._combine_categories(
                df,
                "City_Type",
                "Range_Anxiety_Level",
            )
        )


        # ====================================================
        # 10. Clean infinite values
        # ====================================================

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        return df

class NotebookFeatureEngineer:
    """FE-B + digits + transductive frequencies from Rugved Bane's notebook.

    Reference: kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t
    scriptVersionId=349757944. No labels are used in this class. Frequencies
    intentionally use train + test, matching the competition notebook.
    """

    NUMERIC_COLUMNS = (
        'Age', 'Annual_Income_USD', 'Daily_Commute_km',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level',
    )
    CATEGORICAL_COLUMNS = ('Gender', 'City_Type', 'Current_Car_Type')

    @staticmethod
    def _base_features(df):
        out = df.copy()
        for col in ('Home_Charging_Possible', 'Subsidy_Available'):
            out[col] = out[col].map({'Yes': 1, 'No': 0})
        out['Range_Anxiety_Level'] = out['Range_Anxiety_Level'].map(
            {'Low': 0, 'Medium': 1, 'High': 2}
        )
        out['Home_Charging_x_Subsidy'] = out['Home_Charging_Possible'] * out['Subsidy_Available']
        out['total_charging'] = out['Charging_Stations_Near_Home'] + out['Charging_Stations_Near_Work']
        out['subsidy_x_income'] = out['Subsidy_Available'] * out['Annual_Income_USD']
        out['subsidy_x_concern'] = out['Subsidy_Available'] * out['Environmental_Concern_Level']
        out['income_per_car'] = out['Annual_Income_USD'] / (out['Number_of_Cars_Owned'] + 1)
        out['income_bin'] = pd.cut(
            out['Annual_Income_USD'],
            bins=[0, 30000, 60000, 100000, 200000, float('inf')],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)
        return out

    def transform_pair(self, train, test):
        """Build matching numeric feature matrices; preserve indices/order.

        Select constant/perfectly correlated columns using train only. Keep
        original numeric columns available for downstream target encoding.
        """
        if not train.columns.equals(test.columns):
            raise ValueError('Train and test feature columns must match')
        if {'id', 'Will_Buy_EV'}.intersection(train.columns):
            raise ValueError('Remove id and target before feature engineering')
        # A shared category vocabulary avoids independent get_dummies baselines
        # disagreeing when a category is absent in one dataset.
        combined = pd.concat([train, test], ignore_index=True)
        combined = pd.get_dummies(
            self._base_features(combined),
            columns=list(self.CATEGORICAL_COLUMNS), drop_first=True,
        )
        digits = {}
        for col in self.NUMERIC_COLUMNS:
            for k in range(-4, 4):
                # Keep the notebook's floating-point floor division exactly;
                # rounding first would change its generator-artifact features.
                digits[f'{col}_digit{k}'] = (
                    combined[col].fillna(0) // (10.0 ** k) % 10
                ).astype('int8')
        combined = pd.concat([combined, pd.DataFrame(digits)], axis=1)
        frequencies = {}
        for col in combined.columns:
            keys = combined[col].astype(str)
            frequencies[f'{col}_freq'] = keys.map(
                keys.value_counts(normalize=True)
            ).astype('float32')
        combined = pd.concat([combined, pd.DataFrame(frequencies)], axis=1)
        tr = combined.iloc[:len(train)].copy()
        te = combined.iloc[len(train):].copy()
        tr.index, te.index = train.index, test.index

        protected = set(self.NUMERIC_COLUMNS)
        self.constant_columns_ = [
            c for c in tr if tr[c].nunique() <= 1 and c not in protected
        ]
        tr = tr.drop(columns=self.constant_columns_)
        corr = tr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.correlated_columns_ = [
            c for c in upper if (upper[c] == 1.0).any() and c not in protected
        ]
        self.feature_names_ = tr.drop(columns=self.correlated_columns_).columns.tolist()
        return tr[self.feature_names_], te[self.feature_names_]


class TripleTargetEncoder:
    """Cross-fitted train encodings and outer-train-only validation/test maps.

    Instantiate once per outer fold. Never pass validation/test labels.
    """

    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state

    def fit_transform(self, X, y):
        from sklearn.preprocessing import TargetEncoder
        from sklearn.model_selection import StratifiedKFold

        self.encoders_ = []
        encoded = {}
        columns = list(NotebookFeatureEngineer.NUMERIC_COLUMNS)
        for smooth, name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
            encoder = TargetEncoder(
                smooth=smooth,
                cv=StratifiedKFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state,
                ),
                target_type='binary',
            )
            # fit_transform includes inner CV; fit().transform() would leak.
            values = encoder.fit_transform(X[columns], y)
            self.encoders_.append((name, encoder))
            for j, col in enumerate(columns):
                encoded[f'TE_{col}_{name}'] = values[:, j]
        return pd.concat([X.copy(), pd.DataFrame(encoded, index=X.index)], axis=1)

    def transform(self, X):
        if not hasattr(self, 'encoders_'):
            raise ValueError('Call fit_transform on the outer training fold first')
        encoded = {}
        columns = list(NotebookFeatureEngineer.NUMERIC_COLUMNS)
        for name, encoder in self.encoders_:
            values = encoder.transform(X[columns])
            for j, col in enumerate(columns):
                encoded[f'TE_{col}_{name}'] = values[:, j]
        return pd.concat([X.copy(), pd.DataFrame(encoded, index=X.index)], axis=1)
