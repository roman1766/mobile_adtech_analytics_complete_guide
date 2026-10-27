
import pandas as pd

path = "data/raw/ad_impressions.csv"
df = pd.read_csv(path, parse_dates=["timestamp"])

print("Dataset shape:", df.shape)
print("Duplicate impression IDs:",
      df["impression_id"].duplicated().sum())

print("\nMissing values:")
print(df.isna().sum())

print("\nExperiment group counts:")
print(df.groupby("experiment_group")["user_id"].nunique())

print("\nEvent totals:")
print(df[["clicked", "converted"]].sum())

print("\nInvalid conversions without clicks:")
print(((df["converted"] == 1) & (df["clicked"] == 0)).sum())

assert df["impression_id"].is_unique
assert df["user_id"].nunique() == 20_000
assert df["clicked"].isin([0, 1]).all()
assert df["converted"].isin([0, 1]).all()
assert (df["converted"] <= df["clicked"]).all()
assert df["revenue"].ge(0).all()
assert df["cost"].ge(0).all()

print("\nAll validation checks passed.")
