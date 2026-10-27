
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

DATA_PATH = Path("data/raw/ad_impressions.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

print("\n========== DATA OVERVIEW ==========")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Date range: {df['timestamp'].min()} to "
      f"{df['timestamp'].max()}")

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# --------------------------------------------------
# 3. Check data quality
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")
print("Duplicate impression IDs:",
      df["impression_id"].duplicated().sum())

print("\nMissing values:")
print(df.isna().sum())

print("\nInvalid conversion rows:",
      (df["converted"] > df["clicked"]).sum())

# --------------------------------------------------
# 4. Calculate overall KPIs
# --------------------------------------------------

total_impressions = len(df)
total_clicks = df["clicked"].sum()
total_conversions = df["converted"].sum()
total_cost = df["cost"].sum()
total_revenue = df["revenue"].sum()

overall_ctr = (
    total_clicks / total_impressions
    if total_impressions else 0
)

overall_cvr = (
    total_conversions / total_clicks
    if total_clicks else 0
)

overall_roas = (
    total_revenue / total_cost
    if total_cost else np.nan
)

overall_cpa = (
    total_cost / total_conversions
    if total_conversions else np.nan
)

print("\n========== OVERALL KPIs ==========")
print(f"Impressions: {total_impressions:,.0f}")
print(f"Clicks: {total_clicks:,.0f}")
print(f"Conversions: {total_conversions:,.0f}")
print(f"CTR: {overall_ctr:.2%}")
print(f"Conversion rate: {overall_cvr:.2%}")
print(f"Cost: {total_cost:,.2f}")
print(f"Revenue: {total_revenue:,.2f}")
print(f"ROAS: {overall_roas:.2f}x")
print(f"CPA: {overall_cpa:.2f}")

# --------------------------------------------------
# 5. Campaign performance
# --------------------------------------------------

campaign = (
    df.groupby("campaign_id", as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        cost=("cost", "sum"),
        revenue=("revenue", "sum"),
    )
)

campaign["ctr"] = (
    campaign["clicks"] / campaign["impressions"]
)

campaign["cvr"] = (
    campaign["conversions"]
    / campaign["clicks"].replace(0, np.nan)
)

campaign["roas"] = (
    campaign["revenue"]
    / campaign["cost"].replace(0, np.nan)
)

campaign["cpa"] = (
    campaign["cost"]
    / campaign["conversions"].replace(0, np.nan)
)

campaign = campaign.sort_values("revenue", ascending=False)

print("\n========== CAMPAIGN PERFORMANCE ==========")
print(campaign.round(4).to_string(index=False))

campaign.to_csv(
    REPORT_DIR / "eda_campaign_performance.csv",
    index=False
)

# --------------------------------------------------
# 6. Daily performance trends
# --------------------------------------------------

df["date"] = df["timestamp"].dt.date

daily = (
    df.groupby("date", as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        revenue=("revenue", "sum"),
    )
)

daily["ctr"] = daily["clicks"] / daily["impressions"]

print("\n========== DAILY PERFORMANCE ==========")
print(daily.head(10).to_string(index=False))

daily.to_csv(
    REPORT_DIR / "eda_daily_performance.csv",
    index=False
)

# --------------------------------------------------
# 7. Device analysis
# --------------------------------------------------

device = (
    df.groupby("device", as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        revenue=("revenue", "sum"),
    )
)

device["ctr"] = device["clicks"] / device["impressions"]

print("\n========== DEVICE PERFORMANCE ==========")
print(device.round(4).to_string(index=False))

# --------------------------------------------------
# 8. Interactive visualizations
# --------------------------------------------------

fig_campaign = px.bar(
    campaign,
    x="campaign_id",
    y="revenue",
    color="campaign_id",
    title="Revenue by Campaign",
    labels={
        "campaign_id": "Campaign",
        "revenue": "Revenue",
    },
)
fig_campaign.write_html(
    REPORT_DIR / "revenue_by_campaign.html"
)

fig_daily = px.line(
    daily,
    x="date",
    y="ctr",
    markers=True,
    title="Daily Click-Through Rate",
    labels={
        "date": "Date",
        "ctr": "CTR",
    },
)
fig_daily.update_yaxes(tickformat=".2%")
fig_daily.write_html(
    REPORT_DIR / "daily_ctr.html"
)

fig_device = px.bar(
    device,
    x="device",
    y="ctr",
    color="device",
    title="CTR by Device",
    labels={
        "device": "Device",
        "ctr": "CTR",
    },
)
fig_device.update_yaxes(tickformat=".2%")
fig_device.write_html(
    REPORT_DIR / "ctr_by_device.html"
)

print("\nEDA completed successfully!")
print(f"Reports saved in: {REPORT_DIR.resolve()}")

