
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/raw/ad_impressions.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load data and validate
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

required = [
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

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["user_id"].isna().any():
    raise ValueError("Missing user IDs.")

if df["clicked"].isna().any():
    raise ValueError("Missing target values.")

if not df["clicked"].isin([0, 1]).all():
    raise ValueError("clicked must contain only 0 and 1.")

if df["impression_id"].duplicated().any():
    raise ValueError("Duplicate impression IDs.")

# --------------------------------------------------
# 3. Engineer features
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
groups = df["user_id"]

# --------------------------------------------------
# 4. Split by user
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

assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[test_idx])
)

# --------------------------------------------------
# 5. Build shared preprocessing
# --------------------------------------------------

def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])

# --------------------------------------------------
# 6. Define the models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
}

# --------------------------------------------------
# 7. Train and compare
# --------------------------------------------------

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    fitted_models[name] = model

    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "test_click_rate": y_test.mean(),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
        "log_loss": log_loss(
            y_test, probabilities
        ),
        "brier_score": brier_score_loss(
            y_test, probabilities
        ),
    })

comparison = pd.DataFrame(results).sort_values(
    "pr_auc", ascending=False
)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(5).to_string(index=False))

# --------------------------------------------------
# 8. Calculate permutation importance
# --------------------------------------------------
# Use a subset of test rows to reduce computation.
# Importance is measured by the drop in PR-AUC
# when a feature is randomly shuffled.

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(5000, len(X_test))
sample_positions = rng.choice(
    len(X_test),
    size=sample_size,
    replace=False,
)

X_sample = X_test.iloc[sample_positions]
y_sample = y_test.iloc[sample_positions]

importance_model_name = comparison.iloc[0]["model"]
best_model = fitted_models[importance_model_name]

print(
    f"\nCalculating feature importance for "
    f"{importance_model_name}..."
)

importance = permutation_importance(
    best_model,
    X_sample,
    y_sample,
    scoring="average_precision",
    n_repeats=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(
    REPORT_DIR / "permutation_importance.csv",
    index=False,
)

print("\n========== PERMUTATION IMPORTANCE ==========")
print(importance_df.round(5).to_string(index=False))

# --------------------------------------------------
# 9. Save the selected model
# --------------------------------------------------

import joblib

joblib.dump(
    best_model,
    Path("models") / "best_click_model.joblib",
)

print(
    f"\nBest model by test PR-AUC: "
    f"{importance_model_name}"
)
print("Saved: models/best_click_model.joblib")
print("Saved: reports/model_comparison.csv")
print("Saved: reports/permutation_importance.csv")

