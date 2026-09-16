import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# Load the training data
train = pd.read_csv("data/train.csv")   
test = pd.read_csv("data/test.csv")

# Separate features and target variable

TARGET = "Will_Buy_EV"

X = train.drop(columns=[TARGET, "id"])
y = train[TARGET]

X_test = test.drop(columns= ["id"])

#find categorical columns

cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

print("catgorial columns:")
print(cat_cols)

#train

X_train, X_val, y_train,y_val = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y,
)

#model

model = CatBoostClassifier(
    iterations = 5000,
    depth = 6,
    learning_rate = 0.05,
    random_seed = 42,
    verbose = 100
)

#train

model.fit(
    X_train,
    y_train,
    cat_features= cat_cols
        )

#validate

val_pred = model.predict(X_val).ravel()

accuracy = accuracy_score(y_val, val_pred)

print(f"Validation accuracy: {accuracy:.5f}")

final_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=100
)

final_model.fit(
    X,
    y,
    cat_features=cat_cols
)

test_pred = final_model.predict(X_test).ravel()

submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": test_pred
})

PREDICTION_DIR = Path("predictions")
PREDICTION_DIR.mkdir(exist_ok=True)

existing = list(PREDICTION_DIR.glob("prediction_v*.csv"))
version = len(existing) + 1

filename = (
    PREDICTION_DIR
    / f"prediction_v{version:03d}_acc_{accuracy:.5f}.csv"
)

submission.to_csv(filename, index=False)

print(f"Saved: {filename}")
