import streamlit as st
import pandas as pd
import plotly.express as px
from src.auditor import load_data, profile_dataset, detect_anomalies, calculate_health_score, get_issues

st.set_page_config(page_title="Data Quality Auditor", page_icon="📊", layout="wide")

st.title("📊 Data Quality Auditor")
st.markdown("Upload any dataset and get an instant quality audit report.")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    with st.spinner("Running audit..."):
        df = load_data(uploaded_file)
        profile = profile_dataset(df)
        anomaly_indices, anomaly_count = detect_anomalies(df)
        score = calculate_health_score(profile, anomaly_count, len(df))
        issues = get_issues(profile, anomaly_count, len(df))

    # Header metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Health Score", f"{score} / 100")
    col2.metric("Total Rows", f"{len(df):,}")
    col3.metric("Total Columns", len(df.columns))
    col4.metric("Anomalies Detected", anomaly_count)

    # Health score color
    if score >= 80:
        st.success(f"✅ Dataset is Healthy — Score: {score}/100")
    elif score >= 60:
        st.warning(f"⚠️ Dataset Needs Attention — Score: {score}/100")
    else:
        st.error(f"🔴 Dataset Has Serious Issues — Score: {score}/100")

    st.markdown("---")

    # Issues table
    st.subheader("🔍 Issues Found")
    if issues:
        issues_df = pd.DataFrame(issues)
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("No issues found!")

    st.markdown("---")

    # Column profiles
    st.subheader("📋 Column Profiles")
    profile_df = pd.DataFrame(profile).T.reset_index()
    profile_df.rename(columns={"index": "column"}, inplace=True)
    st.dataframe(profile_df, use_container_width=True)

    st.markdown("---")

    # Null values chart
    st.subheader("🕳️ Null Values by Column")
    null_data = pd.DataFrame([
        {"column": col, "null_pct": info["null_pct"]}
        for col, info in profile.items()
        if info["null_pct"] > 0
    ])
    if not null_data.empty:
        fig = px.bar(null_data, x="column", y="null_pct",
                     title="% Null Values per Column",
                     color="null_pct",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No null values found in any column!")

    st.markdown("---")

    # Anomaly preview
    if anomaly_count > 0:
        st.subheader("🚨 Anomalous Rows Preview")
        st.markdown(f"Showing first 20 of {anomaly_count} anomalous rows detected by Isolation Forest")
        st.dataframe(df.iloc[anomaly_indices[:20]], use_container_width=True)

    st.markdown("---")

    # Raw data preview
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("👆 Upload a CSV or Excel file above to get started.")
    st.markdown("""
    ### What this tool checks:
    - ✅ Null & missing values per column
    - ✅ Data types and unique value counts
    - ✅ ML-powered anomaly detection (Isolation Forest)
    - ✅ Overall dataset health score (0-100)
    - ✅ Column-level statistics (mean, median, std dev)
    """)