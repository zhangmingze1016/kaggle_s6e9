import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Configuration
# ============================================================

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"

ITERATIONS = 500
DEPTH = 6
LEARNING_RATE = 0.05
RANDOM_SEED = 42
TEST_SIZE = 0.2

PREDICTION_DIR = Path("predictions")


# ============================================================
# 2. Load data
# ============================================================

print("Loading data...")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")


# ============================================================
# 3. Prepare features
# ============================================================

X = train.drop(columns=[TARGET, ID_COLUMN])
y = train[TARGET]

X_test = test.drop(columns=[ID_COLUMN])


# Make sure train and test have exactly the same features
assert X.columns.tolist() == X_test.columns.tolist(), (
    "Train and test feature columns do not match!"
)

print(f"Number of features: {X.shape[1]}")


# ============================================================
# 4. Find categorical features
# ============================================================

cat_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nCategorical columns:")
print(cat_cols)


# ============================================================
# 5. Train / validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"\nTraining samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ============================================================
# 6. Create baseline model
# ============================================================

model = CatBoostClassifier(
    iterations=ITERATIONS,
    depth=DEPTH,
    learning_rate=LEARNING_RATE,
    random_seed=RANDOM_SEED,
    verbose=100
)


# ============================================================
# 7. Train baseline model
# ============================================================

print("\nTraining baseline model...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols
)


# ============================================================
# 8. Validate with ROC AUC
# ============================================================

print("\nClasses:")
print(model.classes_)

# Find which probability column corresponds to "Yes"
yes_index = list(model.classes_).index("Yes")

val_prob = model.predict_proba(X_val)[:, yes_index]

# Convert:
# Yes -> 1
# No  -> 0
y_val_binary = (y_val == "Yes").astype(int)

auc = roc_auc_score(
    y_val_binary,
    val_prob
)

print(f"\nValidation ROC AUC: {auc:.5f}")


# ============================================================
# 9. Train final model using ALL training data
# ============================================================

print("\nTraining final model on all training data...")

final_model = CatBoostClassifier(
    iterations=ITERATIONS,
    depth=DEPTH,
    learning_rate=LEARNING_RATE,
    random_seed=RANDOM_SEED,
    verbose=100
)

final_model.fit(
    X,
    y,
    cat_features=cat_cols
)


# ============================================================
# 10. Predict test probabilities
# ============================================================

yes_index_final = list(final_model.classes_).index("Yes")

test_prob = final_model.predict_proba(X_test)[:, yes_index_final]


# ============================================================
# 11. Create Kaggle prediction
# ============================================================

prediction = pd.DataFrame({
    ID_COLUMN: test[ID_COLUMN],
    TARGET: test_prob
})

print("\nPrediction preview:")
print(prediction.head())


# ============================================================
# 12. Create predictions directory
# ============================================================

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 13. Automatically determine experiment version
# ============================================================

existing_predictions = list(
    PREDICTION_DIR.glob("prediction_v*.csv")
)

version = len(existing_predictions) + 1


# ============================================================
# 14. Save prediction
# ============================================================

filename = (
    PREDICTION_DIR
    / f"prediction_v{version:03d}_auc_{auc:.5f}.csv"
)

prediction.to_csv(
    filename,
    index=False
)

print(f"\nSaved prediction: {filename}")
print(f"Local ROC AUC: {auc:.5f}")
print(f"Experiment version: v{version:03d}")