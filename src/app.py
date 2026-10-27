
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mobile Ad-Tech Analytics",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/ad_impressions.csv"
REPORT_DIR = ROOT / "reports"

st.title("Mobile Ad-Tech Analytics")
st.caption(
    "Campaign performance | Experimentation | Machine learning"
)

# --------------------------------------------------
# 2. Load data and model
# --------------------------------------------------

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path, parse_dates=["timestamp"])
    data["date"] = data["timestamp"].dt.date
    return data


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime,
)

required_columns = {
    "impression_id", "timestamp", "user_id",
    "campaign_id", "experiment_group", "device",
    "country", "placement", "bid_price",
    "clicked", "converted", "revenue", "cost",
}

missing = required_columns - set(df.columns)

if missing:
    st.error(f"Missing dataset columns: {sorted(missing)}")
    st.stop()

if df.empty:
    st.warning("The dataset is empty.")
    st.stop()

# --------------------------------------------------
# 3. Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

countries = sorted(df["country"].dropna().unique())
devices = sorted(df["device"].dropna().unique())
campaigns = sorted(df["campaign_id"].dropna().unique())
groups = sorted(df["experiment_group"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries,
)

selected_devices = st.sidebar.multiselect(
    "Device",
    devices,
    default=devices,
)

selected_campaigns = st.sidebar.multiselect(
    "Campaign",
    campaigns,
    default=campaigns,
)

selected_groups = st.sidebar.multiselect(
    "Experiment group",
    groups,
    default=groups,
)

filtered = df[
    df["country"].isin(selected_countries)
    & df["device"].isin(selected_devices)
    & df["campaign_id"].isin(selected_campaigns)
    & df["experiment_group"].isin(selected_groups)
].copy()

if filtered.empty:
    st.warning("No impressions match the selected filters.")
    st.stop()

# --------------------------------------------------
# 4. KPI calculations
# --------------------------------------------------

impressions = len(filtered)
clicks = int(filtered["clicked"].sum())
conversions = int(filtered["converted"].sum())
cost = float(filtered["cost"].sum())
revenue = float(filtered["revenue"].sum())

ctr = clicks / impressions if impressions else 0
cvr = conversions / clicks if clicks else 0
roas = revenue / cost if cost else np.nan
cpa = cost / conversions if conversions else np.nan

st.subheader("Performance Overview")

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Impressions", f"{impressions:,}")
k2.metric("Clicks", f"{clicks:,}")
k3.metric("CTR", f"{ctr:.2%}")
k4.metric("Conversions", f"{conversions:,}")
k5.metric("Revenue", f"{revenue:,.2f}")
k6.metric(
    "ROAS",
    f"{roas:.2f}x" if np.isfinite(roas) else "N/A",
)

st.caption(
    f"Conversion rate: {cvr:.2%} | "
    f"CPA: {cpa:,.2f}" if np.isfinite(cpa)
    else f"Conversion rate: {cvr:.2%} | CPA: N/A"
)

# --------------------------------------------------
# 5. Daily trends
# --------------------------------------------------

st.subheader("Daily Performance")

daily = (
    filtered.groupby("date", as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        revenue=("revenue", "sum"),
    )
)

daily["ctr"] = (
    daily["clicks"] / daily["impressions"]
)

daily["cvr"] = (
    daily["conversions"]
    / daily["clicks"].replace(0, np.nan)
)

left, right = st.columns(2)

with left:
    fig = px.line(
        daily,
        x="date",
        y="impressions",
        markers=True,
        title="Daily Impressions",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(
        daily,
        x="date",
        y="ctr",
        markers=True,
        title="Daily Click-Through Rate",
    )
    fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 6. Campaign performance
# --------------------------------------------------

st.subheader("Campaign Performance")

campaign_perf = (
    filtered.groupby("campaign_id", as_index=False)
    .agg(
        impressions=("impression_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        cost=("cost", "sum"),
        revenue=("revenue", "sum"),
    )
)

campaign_perf["ctr"] = (
    campaign_perf["clicks"]
    / campaign_perf["impressions"]
)

campaign_perf["cvr"] = (
    campaign_perf["conversions"]
    / campaign_perf["clicks"].replace(0, np.nan)
)

campaign_perf["roas"] = (
    campaign_perf["revenue"]
    / campaign_perf["cost"].replace(0, np.nan)
)

campaign_perf["cpa"] = (
    campaign_perf["cost"]
    / campaign_perf["conversions"].replace(0, np.nan)
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        campaign_perf.sort_values("revenue", ascending=False),
        x="campaign_id",
        y="revenue",
        color="campaign_id",
        title="Revenue by Campaign",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        campaign_perf,
        x="campaign_id",
        y="ctr",
        color="campaign_id",
        title="CTR by Campaign",
    )
    fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    campaign_perf.style.format({
        "ctr": "{:.2%}",
        "cvr": "{:.2%}",
        "roas": "{:.2f}x",
        "cpa": "{:.2f}",
        "cost": "{:,.2f}",
        "revenue": "{:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download campaign report",
    data=campaign_perf.to_csv(index=False).encode("utf-8"),
    file_name="campaign_performance.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 7. Device and placement analysis
# --------------------------------------------------

st.subheader("Device and Placement Analysis")

left, right = st.columns(2)

with left:
    device_perf = (
        filtered.groupby("device", as_index=False)
        .agg(
            impressions=("impression_id", "count"),
            clicks=("clicked", "sum"),
        )
    )
    device_perf["ctr"] = (
        device_perf["clicks"] / device_perf["impressions"]
    )

    fig = px.bar(
        device_perf,
        x="device",
        y="ctr",
        color="device",
        title="CTR by Device",
    )
    fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    placement_perf = (
        filtered.groupby("placement", as_index=False)
        .agg(
            impressions=("impression_id", "count"),
            clicks=("clicked", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    placement_perf["ctr"] = (
        placement_perf["clicks"]
        / placement_perf["impressions"]
    )

    fig = px.bar(
        placement_perf,
        x="placement",
        y="ctr",
        color="placement",
        title="CTR by Placement",
    )
    fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 8. A/B test report
# --------------------------------------------------

st.subheader("A/B Test Results")

ab_path = REPORT_DIR / "ab_test_results.csv"

if ab_path.exists():
    ab = pd.read_csv(ab_path)

    if not ab.empty:
        row = ab.iloc[0]

        left, right, third = st.columns(3)

        left.metric(
            "Absolute CTR difference",
            f"{row['absolute_ctr_difference']:.3%}",
        )
        right.metric(
            "Relative CTR uplift",
            f"{row['relative_ctr_uplift']:.2%}",
        )
        third.metric(
            "P-value",
            f"{row['p_value']:.4g}",
        )

        st.write(
            "95% confidence interval for the absolute difference: "
            f"{row['confidence_interval_lower']:.3%} to "
            f"{row['confidence_interval_upper']:.3%}"
        )

        st.caption(
            "This report is the previously calculated experiment result. "
            "The dashboard filters do not recalculate statistical "
            "significance. Interpret it only for the population and "
            "analysis used when the report was generated."
        )
else:
    st.info(
        "Run python src/ab_test.py to generate the A/B test report."
    )

# --------------------------------------------------
# 9. Machine-learning evaluation
# --------------------------------------------------

st.subheader("Machine-Learning Model Evaluation")

metrics_path = REPORT_DIR / "model_comparison.csv"

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)

    if not metrics.empty:
        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            metrics,
            x="model",
            y="pr_auc",
            color="model",
            title="Model Comparison — PR-AUC",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "Run python src/compare_models.py to generate model metrics."
    )

# --------------------------------------------------
# 10. Interactive click prediction
# --------------------------------------------------

st.subheader("Predict Click Probability")

model_path = ROOT / "models/best_click_model.joblib"

if not model_path.exists():
    model_path = ROOT / "models/click_prediction_model.joblib"

if model_path.exists():
    model = load_model(
        str(model_path),
        model_path.stat().st_mtime,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            campaign_id = st.selectbox(
                "Campaign",
                campaigns,
            )
            device = st.selectbox(
                "Device",
                devices,
            )
            country = st.selectbox(
                "Country",
                countries,
            )

        with col2:
            placement_options = sorted(
                df["placement"].dropna().unique()
            )
            placement = st.selectbox(
                "Placement",
                placement_options,
            )
            experiment_group = st.selectbox(
                "Experiment group",
                groups,
            )
            bid_price = st.number_input(
                "Bid price",
                min_value=0.0,
                value=float(df["bid_price"].median()),
            )

        hour = st.slider("Hour of day", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of week (Monday = 0)",
            list(range(7)),
        )

        submitted = st.form_submit_button(
            "Predict click probability"
        )

    if submitted:
        prediction_input = pd.DataFrame([{
            "campaign_id": campaign_id,
            "device": device,
            "country": country,
            "placement": placement,
            "experiment_group": experiment_group,
            "bid_price": bid_price,
            "hour": hour,
            "day_of_week": day_of_week,
        }])

        try:
            probability = model.predict_proba(
                prediction_input
            )[0, 1]

            st.metric(
                "Predicted click probability",
                f"{probability:.2%}",
            )

            st.caption(
                "This is a model estimate based on synthetic data, "
                "not a guarantee or a validated production forecast."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info(
        "Train a model first using "
        "python src/train_model.py or "
        "python src/compare_models.py."
    )

# --------------------------------------------------
# 11. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Portfolio project | Synthetic data | "
    "Python, pandas, Plotly, scikit-learn and Streamlit"
)

