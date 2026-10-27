
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility: the same seed produces the same dataset.
rng = np.random.default_rng(42)

# Dataset size
N_USERS = 20_000
IMPRESSIONS_PER_USER = 5

n = N_USERS * IMPRESSIONS_PER_USER

# Create user-level attributes and randomized assignments.
users = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(N_USERS)],
    "experiment_group": rng.choice(
        ["control", "treatment"],
        size=N_USERS
    ),
    "device": rng.choice(
        ["Android", "iOS"],
        size=N_USERS,
        p=[0.65, 0.35]
    ),
    "country": rng.choice(
        ["UAE", "KSA", "UK", "US"],
        size=N_USERS,
        p=[0.35, 0.25, 0.20, 0.20]
    )
})

# Repeat each user's attributes for their five impressions.
df = users.loc[
    users.index.repeat(IMPRESSIONS_PER_USER)
].reset_index(drop=True)

df["impression_id"] = [
    f"I{i:07d}" for i in range(n)
]

# Generate timestamps over a 30-day period.
start = pd.Timestamp("2026-01-01", tz="UTC")
seconds = rng.integers(0, 30 * 24 * 60 * 60, size=n)

df["timestamp"] = (
    start + pd.to_timedelta(seconds, unit="s")
)

# Assign campaign and placement to each impression.
df["campaign_id"] = rng.choice(
    ["CAMP_01", "CAMP_02", "CAMP_03", "CAMP_04"],
    size=n
)

df["placement"] = rng.choice(
    ["banner", "interstitial", "rewarded_video"],
    size=n,
    p=[0.45, 0.30, 0.25]
)

df["ad_format"] = np.where(
    df["experiment_group"] == "treatment",
    "new_format",
    "existing_format"
)

# Illustrative cost per impression.
df["bid_price"] = np.round(
    rng.uniform(0.002, 0.020, size=n), 5
)

# Simulate click probability.
# These are synthetic assumptions, not industry benchmarks.
click_probability = (
    0.035
    + 0.010 * (df["experiment_group"] == "treatment")
    + 0.008 * (df["device"] == "iOS")
    + 0.015 * (df["placement"] == "rewarded_video")
    + 0.005 * (df["campaign_id"] == "CAMP_01")
)

click_probability = np.clip(
    click_probability, 0.001, 0.90
)

df["clicked"] = rng.binomial(
    1, click_probability
)

# Conversion can happen only after a click in this simulation.
conversion_probability = (
    0.08
    + 0.015 * (df["experiment_group"] == "treatment")
    + 0.010 * (df["campaign_id"] == "CAMP_02")
)

conversion_probability = np.clip(
    conversion_probability, 0.001, 0.90
)

df["converted"] = (
    df["clicked"]
    * rng.binomial(1, conversion_probability)
)

# Revenue is generated for converted impressions.
df["revenue"] = np.where(
    df["converted"] == 1,
    rng.uniform(5, 30, size=n),
    0.0
)

df["cost"] = df["bid_price"]

# Round monetary values for readability.
df["revenue"] = df["revenue"].round(2)
df["cost"] = df["cost"].round(5)

# Sort by event time.
df = df.sort_values("timestamp").reset_index(drop=True)

# Save the raw dataset.
output_path = Path("data/raw/ad_impressions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset created successfully.")
print(f"Rows: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")
print(df.head())
