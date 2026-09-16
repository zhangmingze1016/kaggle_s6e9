from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils.data_loader import EVDataLoader
from utils.prediction_saver import PredictionSaver


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.2
RANDOM_SEED = 42

MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": 100
}


# ============================================================
# Load data
# ============================================================

loader = EVDataLoader()

X, y, X_test, test, cat_cols = loader.load()

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Categorical columns: {cat_cols}")


# ============================================================
# Train / Validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)


# ============================================================
# Train baseline model
# ============================================================

model = CatBoostClassifier(**MODEL_PARAMS)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols
)


# ============================================================
# Validation
# ============================================================

print("\nClasses:")
print(model.classes_)

yes_index = list(model.classes_).index("Yes")

val_prob = model.predict_proba(X_val)[:, yes_index]

y_val_binary = (y_val == "Yes").astype(int)

auc = roc_auc_score(
    y_val_binary,
    val_prob
)

print(f"\nValidation ROC AUC: {auc:.5f}")


# ============================================================
# Train final model using all training data
# ============================================================

final_model = CatBoostClassifier(**MODEL_PARAMS)

final_model.fit(
    X,
    y,
    cat_features=cat_cols
)


# ============================================================
# Predict test
# ============================================================

yes_index = list(final_model.classes_).index("Yes")

test_prob = final_model.predict_proba(
    X_test
)[:, yes_index]


# ============================================================
# Save prediction
# ============================================================

saver = PredictionSaver()

filename = saver.save(
    ids=test["id"],
    predictions=test_prob,
    score=auc
)

print(f"\nLocal ROC AUC: {auc:.5f}")
print(f"Prediction file: {filename}")