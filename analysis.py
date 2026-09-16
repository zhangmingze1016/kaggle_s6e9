import pandas as pd

# First 1000 rows: inspect the structure quickly
train = pd.read_csv("data/train.csv")

print("Train shape:", train.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nData types:")
print(train.dtypes)

# Check test and submission structure
sample = pd.read_csv("data/sample_submission.csv")
test = pd.read_csv("data/test.csv")

print("\nTest shape:", test.shape)
print("Sample submission shape:", sample.shape)

print("\nTest columns:")
print(test.columns.tolist())

print("\nSample submission columns:")
print(sample.columns.tolist())