# Kaggle Playground Series S6E9

Machine learning project for the **Kaggle Playground Series - Season 6, Episode 9** classification competition.

The objective is to predict the probability that a customer will purchase an electric vehicle (EV).

- **Problem:** Binary Classification
- **Target:** `Will_Buy_EV`
- **Evaluation Metric:** ROC AUC
- **Primary Model:** CatBoost

This repository contains the complete machine learning workflow used for the competition, including data loading, model training, cross-validation, hyperparameter experiments, out-of-fold evaluation, experiment tracking, and Kaggle submission generation.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/zhangmingze1016/kaggle_s6e9.git
cd kaggle_s6e9
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The competition dataset is included in the `data/` directory, so no additional data download is required.

---

# Usage

All commands should be executed from the project root directory.

## Run the CatBoost baseline

```bash
python -m training.baseline
```

## Run CatBoost 5-fold cross-validation

```bash
python -m training.catboost_cv
```

The training script supports command-line hyperparameter overrides.

For example:

```bash
python -m training.catboost_cv --depth 5
```

or:

```bash
python -m training.catboost_cv \
    --depth 4 \
    --learning-rate 0.05 \
    --iterations 5000
```

This allows experiments to be performed without modifying the training source code.

Generated predictions are automatically saved to:

```text
predictions/
```

---

# Input

The input datasets are stored in:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Training Data

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

The `id` column is preserved for submission generation but excluded from model training.

## Test Data

`test.csv` contains the same model features as the training data but does not contain `Will_Buy_EV`.

The model predicts:

```text
P(Will_Buy_EV = Yes)
```

for every observation in the test set.

---

# Output

Prediction files are automatically generated in:

```text
predictions/
```

Each prediction file is directly compatible with Kaggle submission.

Example:

```csv
id,Will_Buy_EV
668665,0.009810
668666,0.022302
668667,0.005163
```

`Will_Buy_EV` contains a probability rather than a `Yes` or `No` class prediction because the competition is evaluated using ROC AUC.

---

# Prediction Filename Convention

Prediction filenames contain the most important experiment information so that Kaggle submissions can be identified directly from the filename.

## Format

```text
prediction_vXXX_cb_dX_lrX_iterX_auc_X.csv
```

Example:

```text
prediction_v005_cb_d5_lr0.05_iter3000_auc_0.94190.csv
```

The components represent:

| Component | Meaning | Example |
|---|---|---|
| `v005` | Experiment / prediction version | Version 5 |
| `cb` | Model type | CatBoost |
| `d5` | Tree depth | `depth = 5` |
| `lr0.05` | Learning rate | `learning_rate = 0.05` |
| `iter3000` | Maximum boosting iterations | `iterations = 3000` |
| `auc_0.94190` | Local out-of-fold ROC AUC | `OOF AUC = 0.94190` |

Therefore:

```text
prediction_v005_cb_d5_lr0.05_iter3000_auc_0.94190.csv
```

represents:

```text
Version:        v005
Model:          CatBoost
Depth:          5
Learning Rate:  0.05
Max Iterations: 3000
OOF ROC AUC:    0.94190
```

## Important Note About Iterations

`iter3000` represents the **maximum allowed number of boosting iterations**, not necessarily the number of trees actually used.

Each cross-validation fold uses early stopping independently.

For example, the depth-5 experiment produced:

```text
Fold 1: 2650
Fold 2: 2247
Fold 3: 2389
Fold 4: 2142
Fold 5: 2528

Mean best iteration: 2391
```

while the configured maximum was:

```text
iterations = 3000
```

Detailed experiment parameters and validation results can also be recorded separately for experiment tracking.

---

# Project Structure

```text
kaggle_s6e9/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── predictions/
│   └── prediction_vXXX_cb_dX_lrX_iterX_auc_X.csv
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

## `data/`

Contains the competition input datasets.

## `training/`

Contains model training and validation scripts.

Current experiments:

- `baseline.py` — initial CatBoost baseline using a single stratified train-validation split
- `catboost_cv.py` — main CatBoost training pipeline using stratified 5-fold cross-validation

Future model experiments can also be added to this directory.

## `utils/`

Contains reusable components shared across model experiments.

Current utilities:

- `EVDataLoader` — shared data loading and preparation
- `PredictionSaver` — standardized prediction generation and experiment-aware filenames

## `predictions/`

Contains generated Kaggle submission files.

Prediction filenames contain the experiment version, model, major hyperparameters, and local OOF ROC AUC so submissions can be identified easily on the Kaggle submission page.

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
7. Verifying that train and test features match
8. Returning the prepared datasets to the training pipeline

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

Centralizing data loading ensures that different model experiments use the same input pipeline.

---

# Prediction Pipeline

Prediction output is handled by:

```text
utils/prediction_saver.py
```

The `PredictionSaver` class:

1. Creates the `predictions/` directory if necessary
2. Determines the next experiment version
3. Creates a Kaggle-compatible prediction DataFrame
4. Adds important experiment parameters to the filename
5. Adds the local OOF ROC AUC to the filename
6. Saves the prediction as a CSV file

Example:

```text
prediction_v005_cb_d5_lr0.05_iter3000_auc_0.94190.csv
```

This allows a Kaggle submission to be associated directly with the model configuration that produced it.

---

# Baseline Model

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

The initial baseline configuration was approximately:

```python
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
}
```

Initial result:

```text
Local ROC AUC: 0.94110
Public LB:     0.94083
```

This serves as the initial benchmark.

---

# Cross-Validation

The main training pipeline uses:

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

# Early Stopping

Each fold is trained independently with early stopping.

A high maximum value for `iterations` can therefore be specified without requiring every model to use all iterations.

Conceptually:

```text
Maximum iterations
        │
        ▼
    CatBoost
        │
        ▼
Validation AUC monitored
        │
        ▼
No improvement for configured patience
        │
        ▼
Training stops
```

Different folds may therefore have different best iterations.

For the depth-5 experiment:

```text
Fold 1: 2650
Fold 2: 2247
Fold 3: 2389
Fold 4: 2142
Fold 5: 2528

Mean:   2391
```

---

# Out-of-Fold Evaluation

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

The resulting OOF ROC AUC is the primary local metric used to compare experiments.

---

# Test Prediction

Each fold model also predicts the competition test set.

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

# Hyperparameter Experiments

The CatBoost training pipeline supports controlled hyperparameter experiments.

Important parameters currently being investigated include:

```text
depth
learning_rate
iterations
```

Only selected parameters are changed between experiments while the cross-validation folds and random seed remain fixed.

For example:

```bash
python -m training.catboost_cv --depth 5
```

can be compared against:

```bash
python -m training.catboost_cv --depth 6
```

using exactly the same folds.

The current depth experiments show that reducing depth from 6 to 5 slightly improved OOF ROC AUC.

The depth-5 fold results were:

```text
Fold 1: 0.94074
Fold 2: 0.94156
Fold 3: 0.94287
Fold 4: 0.94252
Fold 5: 0.94185

Mean Fold AUC: 0.94191
Std Fold AUC:  0.00075
OOF ROC AUC:   0.94190
```

All five folds improved slightly relative to the depth-6 experiment.

Hyperparameter experiments are evaluated primarily using OOF performance rather than repeatedly optimizing against the Kaggle public leaderboard.

---

# Evaluation Metric

The competition uses **ROC AUC**.

ROC AUC evaluates how effectively the model ranks positive observations above negative observations across classification thresholds.

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

The positive class is:

```text
Will_Buy_EV = Yes
```

Therefore model predictions represent:

```text
P(Will_Buy_EV = Yes)
```

---

# Experiment Workflow

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
 ▼
Hyperparameter Experiments
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
             Ensemble
                │
                ▼
        Kaggle Submission
```

Major model changes are evaluated locally using the same cross-validation framework before Kaggle leaderboard performance is considered.

---

# Results

| Experiment | Validation Strategy | Local ROC AUC | Public LB |
|---|---|---:|---:|
| CatBoost Baseline | Stratified 80/20 | 0.94110 | 0.94083 |
| CatBoost CV — depth 6 | 5-Fold Stratified CV | 0.94179 | 0.94164 |
| CatBoost CV — depth 5 | 5-Fold Stratified CV | **0.94190** | Pending |
| CatBoost CV — depth 4 | 5-Fold Stratified CV | In Progress | - |
| LightGBM CV | 5-Fold Stratified CV | Planned | - |
| XGBoost CV | 5-Fold Stratified CV | Planned | - |
| Ensemble | OOF-based | Planned | - |

The current best completed local CatBoost experiment uses:

```text
depth = 5
learning_rate = 0.05
max_iterations = 3000
```

with:

```text
OOF ROC AUC = 0.94190
```

---

# Reproducibility

Random seeds are fixed where applicable:

```python
RANDOM_SEED = 42
```

Cross-validation uses the same shuffled stratified folds so model experiments can be compared under consistent validation conditions.

Command-line hyperparameters and structured prediction filenames make individual experiments easier to reproduce and identify.

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
- [x] Implement CatBoost 5-fold cross-validation
- [x] Implement early stopping
- [x] Add command-line hyperparameter configuration
- [x] Add experiment-aware prediction filenames
- [x] Begin systematic CatBoost hyperparameter experiments
- [ ] Complete CatBoost hyperparameter search
- [ ] Train LightGBM
- [ ] Train XGBoost
- [ ] Perform feature engineering
- [ ] Compare OOF predictions
- [ ] Build model ensembles
- [ ] Select final submissions