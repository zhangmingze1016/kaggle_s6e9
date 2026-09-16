import pandas as pd

class EVDataLoader:

    def __init__(
            self,
            train_path = "data/train.csv",
            test_path = "data/test.csv",
            target = "Will_Buy_EV",
            id_column = "id"
        ):
            self.train_path = train_path
            self.test_path = test_path
            self.target = target
            self.id_column = id_column

    def load(self):

          train = pd.read_csv(self.train_path)
          test = pd.read_csv(self.test_path)

          X = train.drop(
                columns = [self.id_columntarget, self.id_column]
          )

          y = train[self.target]

          X_test = test.drop(columns= [self.id_column])

          assert X.columns.tolist() == X_test.columns.tolist(), \
            "Train and test features do not match"

          cat_cols = X.select_dtypes(
                include=["object", "str"]
          ).columns.tolist

          return X, y, X_test, test, cat_cols

