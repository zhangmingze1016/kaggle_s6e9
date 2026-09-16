# Kaggle Playground Series S6E9

Machine learning project for the Kaggle Playground Series Season 6, Episode 9 competition.

The goal of this project is to predict the probability that a customer will purchase an electric vehicle (EV).

## Competition

**Target:** `Will_Buy_EV`

**Evaluation Metric:** ROC AUC

Because the competition is evaluated using ROC AUC, models output the predicted probability of the positive class (`Yes`) rather than binary class predictions.

---

## Current Results

| Experiment | Validation | Local ROC AUC | Public LB |
|---|---|---:|---:|
| CatBoost Baseline | 80/20 Stratified Split | 0.94110 | 0.94083 |
| CatBoost CV | 5-Fold Stratified CV | In Progress | - |
| LightGBM CV | 5-Fold Stratified CV | Planned | - |
| XGBoost CV | 5-Fold Stratified CV | Planned | - |
| Ensemble | OOF-based | Planned | - |

The initial CatBoost baseline achieved a local ROC AUC of **0.94110** and a Kaggle Public Leaderboard score of **0.94083**.

---

## Project Structure

```text
kaggle_s6e9/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── predictions/
│   └── prediction_vXXX_auc_XXXXX.csv
│
├── training/
│   ├── __init__.py
│   ├── baseline.py
│   └── catboost_cv.py
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── prediction_saver.py
│
├── analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

### `data/`

Contains the competition training data, test data, and sample submission.

### `training/`

Contains model training and validation scripts.

- `baseline.py` — initial CatBoost baseline using a single train-validation split
- `catboost_cv.py` — CatBoost with 5-fold stratified cross-validation

Additional model experiments will be added here.

### `utils/`

Contains reusable components shared by different model experiments.

- `data_loader.py` — centralized data loading and feature preparation
- `prediction_saver.py` — standardized prediction and submission file generation

### `predictions/`

Contains generated test-set predictions.

Prediction files are automatically versioned and include the corresponding local ROC AUC score in the filename.

Example:

```text
prediction_v003_auc_0.94150.csv
```

---

# Methodology

## 1. Data Pipeline

Data loading and basic feature preparation are centralized in the `EVDataLoader` class located in:

```text
utils/data_loader.py
```

The loader is responsible for:

- Loading `train.csv` and `test.csv`
- Separating the target from the training features
- Removing the ID column from model features
- Extracting test IDs for submission generation
- Detecting categorical columns
- Providing a consistent feature set to all models

Conceptually:

```text
train.csv ──┐
            │
            ├──→ EVDataLoader
            │       │
test.csv ───┘       │
                    ├──→ X
                    ├──→ y
                    ├──→ X_test
                    ├──→ test_ids
                    └──→ categorical columns
```

Centralizing this logic ensures that CatBoost, LightGBM, XGBoost, and future models use the same underlying data pipeline.

---

## 2. Baseline

The first model is a CatBoost classifier.

The training data is divided using a stratified 80/20 train-validation split:

```text
Full Training Data
        │
        ├── 80% Training
        │
        └── 20% Validation
```

Stratification preserves approximately the same target-class distribution in both subsets.

The baseline model configuration is:

```python
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
}
```

The baseline achieved:

```text
Local ROC AUC: 0.94110
Public LB:     0.94083
```

This provides the initial benchmark against which future experiments can be compared.

---

## 3. Cross-Validation

A single train-validation split can be sensitive to the particular observations assigned to the validation set.

To obtain a more reliable estimate of model performance, the next experiment uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The training data is divided into five folds:

```text
Fold 1: [Validation] [Train]      [Train]      [Train]      [Train]
Fold 2: [Train]      [Validation] [Train]      [Train]      [Train]
Fold 3: [Train]      [Train]      [Validation] [Train]      [Train]
Fold 4: [Train]      [Train]      [Train]      [Validation] [Train]
Fold 5: [Train]      [Train]      [Train]      [Train]      [Validation]
```

Each observation:

- is used for training in four folds
- is used for validation exactly once

This allows every training observation to receive an out-of-fold prediction.

---

## 4. Out-of-Fold Predictions

For every fold, the model predicts probabilities for observations that were not used to train that model.

These predictions are stored in their original positions:

```text
Fold 1 validation predictions ──┐
Fold 2 validation predictions ──┤
Fold 3 validation predictions ──┼──→ Complete OOF Predictions
Fold 4 validation predictions ──┤
Fold 5 validation predictions ──┘
```

The final OOF ROC AUC is calculated using all out-of-fold predictions.

This provides a more reliable local estimate of generalization performance than a single train-validation split.

---

## 5. Test Prediction

Each of the five cross-validation models also predicts probabilities for the competition test set.

For each test observation, the final prediction is the average of the five model predictions:

```text
Model 1 ──→ Test Prediction 1 ──┐
Model 2 ──→ Test Prediction 2 ──┤
Model 3 ──→ Test Prediction 3 ──┼──→ Average ──→ Final Prediction
Model 4 ──→ Test Prediction 4 ──┤
Model 5 ──→ Test Prediction 5 ──┘
```

This reduces dependence on any single training split.

---

## 6. Prediction Pipeline

Prediction files are generated through the `PredictionSaver` class located in:

```text
utils/prediction_saver.py
```

The saver:

- Creates the prediction directory when necessary
- Builds a Kaggle-compatible submission DataFrame
- Automatically assigns an experiment version
- Includes the local ROC AUC in the filename
- Saves the prediction as a CSV file

The naming convention is:

```text
prediction_vXXX_auc_XXXXX.csv
```

For example:

```text
prediction_v001_auc_0.94050.csv
prediction_v002_auc_0.94110.csv
prediction_v003_auc_0.94203.csv
```

This makes it easier to associate each submission with its corresponding local experiment.

---

# Evaluation

The primary evaluation metric is **ROC AUC**.

Unlike accuracy, ROC AUC evaluates how well the model ranks positive observations above negative observations across different classification thresholds.

Therefore, submissions contain probabilities:

```text
id,Will_Buy_EV
668665,0.009810
668666,0.022302
668667,0.005163
...
```

rather than class labels such as:

```text
Yes
No
Yes
```

The positive class is:

```text
Will_Buy_EV = Yes
```

and the submitted prediction represents:

```text
P(Will_Buy_EV = Yes)
```

---

# Experiment Workflow

The development workflow for this project is:

```text
Data Exploration
      │
      ▼
CatBoost Baseline
      │
      ▼
5-Fold Cross-Validation
      │
      ▼
OOF Evaluation
      │
      ├─────────────┐
      ▼             ▼
  CatBoost       LightGBM
      │             │
      └──────┬──────┘
             │
             ▼
          XGBoost
             │
             ▼
     Feature Engineering
             │
             ▼
        Model Tuning
             │
             ▼
          Ensemble
             │
             ▼
     Kaggle Submission
```

The main principle is to evaluate model changes using the same cross-validation framework before relying on leaderboard results.

---

# Planned Experiments

Future experiments include:

- Complete CatBoost 5-fold CV baseline
- LightGBM cross-validation
- XGBoost cross-validation
- Hyperparameter experiments
- Feature engineering
- Feature importance analysis
- Comparison of OOF predictions
- Probability averaging
- Rank averaging
- Multi-model ensemble

---

# Running the Project

Create and activate the virtual environment before running experiments.

Run the baseline from the project root:

```bash
python -m training.baseline
```

Run CatBoost cross-validation:

```bash
python -m training.catboost_cv
```

Running scripts as modules ensures that project-level packages such as `utils` can be imported correctly.

For example:

```python
from utils.data_loader import EVDataLoader
from utils.prediction_saver import PredictionSaver
```

---

# Dependencies

Main libraries currently used:

```text
pandas
numpy
scikit-learn
catboost
```

Install all project dependencies with:

```bash
pip install -r requirements.txt
```

---

# Reproducibility

Random seeds are fixed where applicable:

```python
RANDOM_SEED = 42
```

Cross-validation uses shuffled stratified folds with the same random seed so that experiments can be compared using identical splits.

As new models are introduced, the same validation strategy will be used whenever possible to make local model comparisons meaningful.

---

# Current Status

- [x] Download and inspect competition data
- [x] Build CatBoost baseline
- [x] Switch evaluation from accuracy to ROC AUC
- [x] Generate probability-based submissions
- [x] Submit baseline to Kaggle
- [x] Refactor data loading into `EVDataLoader`
- [x] Refactor prediction output into `PredictionSaver`
- [x] Set up project package structure
- [ ] Complete CatBoost 5-fold CV
- [ ] Train LightGBM model
- [ ] Train XGBoost model
- [ ] Feature engineering
- [ ] Hyperparameter tuning
- [ ] Model ensemble

Current best Public Leaderboard score:

```text
0.94083
```