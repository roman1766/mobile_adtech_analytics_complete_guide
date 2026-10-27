
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/raw/ad_impressions.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

required_columns = [
    "user_id",
    "impression_id",
    "timestamp",
    "campaign_id",
    "device",
    "country",
    "placement",
    "bid_price",
    "experiment_group",
    "clicked",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["user_id"].isna().any():
    raise ValueError("Missing user_id values.")

if df["clicked"].isna().any():
    raise ValueError("Missing target values.")

if not df["clicked"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if df["impression_id"].duplicated().any():
    raise ValueError("Duplicate impression IDs detected.")

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

categorical_features = [
    "campaign_id",
    "device",
    "country",
    "placement",
    "experiment_group",
]

numeric_features = [
    "bid_price",
    "hour",
    "day_of_week",
]

feature_columns = categorical_features + numeric_features

X = df[feature_columns].copy()
y = df["clicked"].astype(int)

# Each user must stay entirely in one split.
groups = df["user_id"]

# --------------------------------------------------
# 4. Split by user to reduce user-level leakage
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_users = set(groups.iloc[train_idx])
test_users = set(groups.iloc[test_idx])

assert train_users.isdisjoint(test_users)

print("========== DATA SPLIT ==========")
print(f"Training impressions: {len(X_train):,}")
print(f"Test impressions: {len(X_test):,}")
print(f"Training users: {len(train_users):,}")
print(f"Test users: {len(test_users):,}")
print(f"Training click rate: {y_train.mean():.3%}")
print(f"Test click rate: {y_test.mean():.3%}")

# --------------------------------------------------
# 5. Preprocess features
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])

# --------------------------------------------------
# 6. Create the model pipeline
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    ),
])

# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 8. Predict click probabilities
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

pr_auc = average_precision_score(
    y_test, probabilities
)

roc_auc = roc_auc_score(
    y_test, probabilities
)

test_log_loss = log_loss(
    y_test, probabilities
)

brier = brier_score_loss(
    y_test, probabilities
)

print("\n========== MODEL EVALUATION ==========")
print(f"Test click rate: {y_test.mean():.4%}")
print(f"PR-AUC / Average Precision: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Log Loss: {test_log_loss:.5f}")
print(f"Brier Score: {brier:.5f}")

# --------------------------------------------------
# 10. Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "actual_clicked": y_test.to_numpy(),
    "predicted_click_probability": probabilities,
})

predictions.to_csv(
    REPORT_DIR / "click_predictions.csv",
    index=False,
)

# --------------------------------------------------
# 11. Save evaluation metrics
# --------------------------------------------------

metrics = pd.DataFrame([{
    "test_click_rate": y_test.mean(),
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "log_loss": test_log_loss,
    "brier_score": brier,
}])

metrics.to_csv(
    REPORT_DIR / "model_metrics.csv",
    index=False,
)

# --------------------------------------------------
# 12. Save trained model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "click_prediction_model.joblib",
)

print("\nModel saved to:")
print((MODEL_DIR / "click_prediction_model.joblib").resolve())

print("\nPredictions saved to reports/click_predictions.csv")
print("Metrics saved to reports/model_metrics.csv")

