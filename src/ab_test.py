
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/raw/ad_impressions.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = {
    "user_id",
    "experiment_group",
    "impression_id",
    "clicked",
    "converted",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["user_id"].isna().any():
    raise ValueError("user_id contains missing values.")

if df["experiment_group"].isna().any():
    raise ValueError("experiment_group contains missing values.")

if df["impression_id"].duplicated().any():
    raise ValueError("Duplicate impression IDs detected.")

if set(df["experiment_group"].unique()) != {
    "control", "treatment"
}:
    raise ValueError(
        "Expected exactly control and treatment groups."
    )

if not df["clicked"].isin([0, 1]).all():
    raise ValueError("clicked must contain only 0 or 1.")

if not df["converted"].isin([0, 1]).all():
    raise ValueError("converted must contain only 0 or 1.")

if (df["converted"] > df["clicked"]).any():
    raise ValueError(
        "A conversion exists without a corresponding click."
    )

# --------------------------------------------------
# 3. Verify user-level group assignment
# --------------------------------------------------

groups_per_user = (
    df.groupby("user_id")["experiment_group"].nunique()
)

if (groups_per_user > 1).any():
    raise ValueError(
        "Some users appear in both experiment groups."
    )

print("Data validation passed.")

# --------------------------------------------------
# 4. Aggregate impressions to users
# --------------------------------------------------

users = (
    df.groupby(["user_id", "experiment_group"], as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
    )
)

users["user_ctr"] = (
    users["clicks"] / users["impressions"]
)

users["user_conversion_rate"] = (
    users["conversions"] / users["impressions"]
)

# --------------------------------------------------
# 5. Summarize the groups
# --------------------------------------------------

summary = (
    users.groupby("experiment_group")
    .agg(
        users=("user_id", "nunique"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        mean_user_ctr=("user_ctr", "mean"),
        mean_user_conversion_rate=(
            "user_conversion_rate", "mean"
        ),
    )
)

summary["pooled_ctr"] = (
    summary["clicks"] / summary["impressions"]
)

summary["pooled_conversion_rate"] = (
    summary["conversions"] / summary["impressions"]
)

print("\n========== EXPERIMENT SUMMARY ==========")
print(summary.round(5).to_string())

# --------------------------------------------------
# 6. Prepare user-level CTR samples
# --------------------------------------------------

control = users.loc[
    users["experiment_group"] == "control",
    "user_ctr"
].to_numpy()

treatment = users.loc[
    users["experiment_group"] == "treatment",
    "user_ctr"
].to_numpy()

if len(control) < 2 or len(treatment) < 2:
    raise ValueError(
        "At least two users per group are required."
    )

control_mean = np.mean(control)
treatment_mean = np.mean(treatment)

difference = treatment_mean - control_mean

relative_uplift = (
    difference / control_mean
    if control_mean != 0
    else np.nan
)

# --------------------------------------------------
# 7. Welch's t-test
# --------------------------------------------------

test = stats.ttest_ind(
    treatment,
    control,
    equal_var=False,
    alternative="two-sided",
)

p_value = float(test.pvalue)

# --------------------------------------------------
# 8. Calculate a 95% confidence interval
#    for treatment minus control
# --------------------------------------------------

control_var = np.var(control, ddof=1)
treatment_var = np.var(treatment, ddof=1)

control_n = len(control)
treatment_n = len(treatment)

standard_error = np.sqrt(
    control_var / control_n
    + treatment_var / treatment_n
)

degrees_of_freedom = (
    (control_var / control_n + treatment_var / treatment_n) ** 2
    / (
        (control_var / control_n) ** 2 / (control_n - 1)
        + (treatment_var / treatment_n) ** 2
        / (treatment_n - 1)
    )
)

critical_value = stats.t.ppf(
    1 - ALPHA / 2,
    df=degrees_of_freedom,
)

ci_lower = difference - critical_value * standard_error
ci_upper = difference + critical_value * standard_error

# --------------------------------------------------
# 9. Report the result
# --------------------------------------------------

print("\n========== PRIMARY A/B TEST: USER CTR ==========")

print(f"Control users: {control_n:,}")
print(f"Treatment users: {treatment_n:,}")

print(f"Control mean user CTR: {control_mean:.4%}")
print(f"Treatment mean user CTR: {treatment_mean:.4%}")

print(f"Absolute difference: {difference:.4%}")
print(f"Relative uplift: {relative_uplift:.2%}")

print(f"Welch t-statistic: {test.statistic:.4f}")
print(f"P-value: {p_value:.6g}")

print(
    f"95% CI for absolute difference: "
    f"[{ci_lower:.4%}, {ci_upper:.4%}]"
)

if p_value < ALPHA:
    print(
        "Statistical result: Reject the null hypothesis "
        "at the 5% significance level."
    )
else:
    print(
        "Statistical result: Do not reject the null "
        "hypothesis at the 5% significance level."
    )

if ci_lower > 0:
    print(
        "Interpretation: The confidence interval indicates "
        "a positive treatment effect on mean user CTR."
    )
elif ci_upper < 0:
    print(
        "Interpretation: The confidence interval indicates "
        "a negative treatment effect on mean user CTR."
    )
else:
    print(
        "Interpretation: The confidence interval includes "
        "zero, so the direction of the effect remains uncertain."
    )

# --------------------------------------------------
# 10. Save reusable reports
# --------------------------------------------------

summary.to_csv(
    REPORT_DIR / "ab_test_group_summary.csv"
)

results = pd.DataFrame([{
    "control_users": control_n,
    "treatment_users": treatment_n,
    "control_mean_user_ctr": control_mean,
    "treatment_mean_user_ctr": treatment_mean,
    "absolute_ctr_difference": difference,
    "relative_ctr_uplift": relative_uplift,
    "p_value": p_value,
    "confidence_interval_lower": ci_lower,
    "confidence_interval_upper": ci_upper,
    "alpha": ALPHA,
}])

results.to_csv(
    REPORT_DIR / "ab_test_results.csv",
    index=False,
)

print("\nA/B testing completed.")
print("Reports saved in the reports/ directory.")

