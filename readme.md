# Kaggle Playground Series S6E9

Machine learning project for the **Kaggle Playground Series - Season 6, Episode 9** classification competition.

The objective is to predict the probability that a customer will purchase an electric vehicle (EV).

- **Problem:** Binary Classification
- **Target:** `Will_Buy_EV`
- **Evaluation Metric:** ROC AUC

This repository contains the data pipeline, model training code, cross-validation framework, and prediction utilities used for the competition.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhangmingze1016/kaggle_s6e9
cd kaggle_competition1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The competition dataset is included in the `data/` directory, so no additional data download is required.

---

## Usage

All commands should be executed from the project root directory.

### Run the CatBoost baseline

```bash
python -m training.baseline
```

### Run CatBoost 5-fold cross-validation

```bash
python -m training.catboost_cv
```

Generated predictions are automatically saved to:

```text
predictions/
```

---

## Input

The input datasets are stored in:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Training Data

`train.csv` contains the following columns:

| Column | Role |
|---|---|
| `id` | Observation identifier |
| `Age` | Feature |
| `Annual_Income_USD` | Feature |
| `Daily_Commute_km` | Feature |
| `Number_of_Cars_Owned` | Feature |
| `Charging_Stations_Near_Home` | Feature |
| `Charging_Stations_Near_Work` | Feature |
| `Environmental_Concern_Level` | Feature |
| `Gender` | Categorical feature |
| `City_Type` | Categorical feature |
| `Current_Car_Type` | Categorical feature |
| `Home_Charging_Possible` | Categorical feature |
| `Subsidy_Available` | Categorical feature |
| `Range_Anxiety_Level` | Categorical feature |
| `Will_Buy_EV` | Target |

The target variable is:

```text
Will_Buy_EV
```

with two classes:

```text
No
Yes
```

The `id` column is preserved for submission generation but is excluded from model training.

### Test Data

`test.csv` contains the same model features as the training data but does not contain `Will_Buy_EV`.

The model predicts:

```text
P(Will_Buy_EV = Yes)
```

for every observation in the test set.

---

## Output

Prediction files are automatically generated in:

```text
predictions/
```

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

Each file contains:

```csv
id,Will_Buy_EV
668665,0.009810
668666,0.022302
668667,0.005163
```

`Will_Buy_EV` contains a probability rather than a `Yes` or `No` class prediction because the competition is evaluated using ROC AUC.

Prediction files are generated locally and are not tracked by Git.

---

## Project Structure

```text
kaggle_competition1/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── predictions/
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
├── requirements.txt
├── .gitignore
└── README.md
```

### `data/`

Contains the competition input datasets.

### `training/`

Contains model training and validation scripts.

Current experiments:

- `baseline.py` — CatBoost with a single stratified train-validation split
- `catboost_cv.py` — CatBoost with 5-fold stratified cross-validation

Future model experiments will also be added to this directory.

### `utils/`

Contains reusable components shared by different model experiments.

Current utilities:

- `EVDataLoader` — shared data loading and preparation
- `PredictionSaver` — standardized prediction file generation

### `predictions/`

Contains generated Kaggle submission files.

This directory is generated during model execution and is not intended to be committed to the repository.

---

# Methodology

## Data Pipeline

All model experiments use the same data-loading pipeline through:

```text
utils/data_loader.py
```

The `EVDataLoader` class is responsible for:

1. Loading the training dataset
2. Loading the test dataset
3. Separating features and target
4. Removing `id` from model features
5. Extracting test IDs
6. Detecting categorical features
7. Returning the prepared datasets to the training pipeline

Conceptually:

```text
                 EVDataLoader
                /            \
               /              \
        train.csv            test.csv
            │                    │
            ▼                    ▼
         X + y                X_test
                                 │
                                 ▼
                              test_ids

                +
        categorical columns
```

Centralizing data loading ensures that different models use the same input pipeline.

---

## Prediction Pipeline

Prediction output is handled by:

```text
utils/prediction_saver.py
```

The `PredictionSaver` class:

1. Creates the `predictions/` directory if necessary
2. Determines the next experiment version
3. Creates a Kaggle-compatible prediction DataFrame
4. Adds the local ROC AUC to the filename
5. Saves the result as a CSV file

For example:

```text
prediction_v003_auc_0.94203.csv
```

This keeps prediction files associated with their corresponding local experiment results.

---

## Baseline Model

The initial baseline uses `CatBoostClassifier`.

The training data is divided using a stratified 80/20 split:

```text
Training Dataset
      │
      ├───────────────┐
      ▼               ▼
  80% Train      20% Validation
      │               │
      └── CatBoost ───┘
              │
              ▼
           ROC AUC
```

The baseline configuration is approximately:

```python
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
}
```

The initial experiment achieved:

```text
Local ROC AUC: 0.94110
Public LB:     0.94083
```

This serves as the initial benchmark for future experiments.

---

## Cross-Validation

A single validation split can produce a noisy estimate of model performance depending on which observations happen to enter the validation set.

The main evaluation pipeline therefore uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The training data is divided into five folds:

```text
          F1    F2    F3    F4    F5

Fold 1    VAL   TR    TR    TR    TR
Fold 2    TR    VAL   TR    TR    TR
Fold 3    TR    TR    VAL   TR    TR
Fold 4    TR    TR    TR    VAL   TR
Fold 5    TR    TR    TR    TR    VAL
```

Each observation is:

- used for training four times
- used for validation exactly once

This produces out-of-fold predictions for the entire training dataset.

---

## Out-of-Fold Evaluation

For each fold:

```text
Training Folds
      │
      ▼
    Model
      │
      ▼
Held-Out Fold
      │
      ▼
Validation Probability
```

The validation probabilities are stored at their original dataset positions.

After all five folds:

```text
Fold 1 predictions ──┐
Fold 2 predictions ──┤
Fold 3 predictions ──┼──► Complete OOF Predictions
Fold 4 predictions ──┤
Fold 5 predictions ──┘
                              │
                              ▼
                           ROC AUC
```

The resulting OOF ROC AUC is used as the primary local metric when comparing experiments.

---

## Test Prediction

Each fold model also predicts the competition test set.

This produces five predictions for every test observation:

```text
Fold 1 Model ──► Test probabilities ──┐
Fold 2 Model ──► Test probabilities ──┤
Fold 3 Model ──► Test probabilities ──┼──► Average
Fold 4 Model ──► Test probabilities ──┤       │
Fold 5 Model ──► Test probabilities ──┘       ▼
                                          Submission
```

The final test probability is the average of the five fold predictions.

This reduces dependence on any single train-validation split.

---

# Evaluation Metric

The competition uses **ROC AUC**.

ROC AUC evaluates how effectively the model ranks positive examples above negative examples across classification thresholds.

Because of this, the model submits probabilities instead of hard class predictions.

For example:

```text
0.95
0.72
0.31
0.04
```

rather than:

```text
Yes
Yes
No
No
```

The positive class for this project is:

```text
Will_Buy_EV = Yes
```

Therefore, model predictions represent:

```text
P(Will_Buy_EV = Yes)
```

---

# Experiment Workflow

The project follows this general experimental workflow:

```text
Data
 │
 ▼
EVDataLoader
 │
 ▼
Baseline
 │
 ▼
5-Fold Cross-Validation
 │
 ▼
OOF Evaluation
 │
 ├──────────────┬──────────────┐
 ▼              ▼              ▼
CatBoost     LightGBM       XGBoost
 │              │              │
 └──────────────┼──────────────┘
                ▼
       Feature Engineering
                │
                ▼
       Hyperparameter Tuning
                │
                ▼
            Ensemble
                │
                ▼
       Kaggle Submission
```

All major model changes should be evaluated locally using the same cross-validation framework before comparing Kaggle leaderboard performance.

---

# Results

| Experiment | Validation Strategy | Local ROC AUC | Public LB |
|---|---|---:|---:|
| CatBoost Baseline | Stratified 80/20 | 0.94110 | 0.94083 |
| CatBoost CV | 5-Fold Stratified CV | In Progress | - |
| LightGBM CV | 5-Fold Stratified CV | Planned | - |
| XGBoost CV | 5-Fold Stratified CV | Planned | - |
| Ensemble | OOF-based | Planned | - |

The results table will be updated as new experiments are completed.

---

# Reproducibility

Random seeds are fixed where applicable:

```python
RANDOM_SEED = 42
```

Cross-validation uses the same shuffled stratified folds so that model experiments can be compared under consistent validation conditions.

---

# Roadmap

- [x] Set up project environment
- [x] Load and inspect competition data
- [x] Build initial CatBoost baseline
- [x] Use probability predictions for ROC AUC
- [x] Generate Kaggle-compatible submissions
- [x] Submit baseline to Kaggle
- [x] Create reusable `EVDataLoader`
- [x] Create reusable `PredictionSaver`
- [x] Refactor project into packages
- [ ] Complete CatBoost 5-fold cross-validation
- [ ] Train LightGBM
- [ ] Train XGBoost
- [ ] Perform feature engineering
- [ ] Tune model hyperparameters
- [ ] Compare OOF predictions
- [ ] Build model ensembles
- [ ] Select final submissions