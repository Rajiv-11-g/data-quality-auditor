import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    else:
        raise ValueError("Only CSV and Excel files supported")

def profile_dataset(df):
    profile = {}
    for col in df.columns:
        col_data = df[col]
        info = {
            "dtype": str(col_data.dtype),
            "total": len(col_data),
            "nulls": int(col_data.isnull().sum()),
            "null_pct": round(col_data.isnull().mean() * 100, 2),
            "unique": int(col_data.nunique()),
            "unique_pct": round(col_data.nunique() / len(col_data) * 100, 2),
        }
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                "mean": round(col_data.mean(), 2),
                "median": round(col_data.median(), 2),
                "std": round(col_data.std(), 2),
                "min": round(col_data.min(), 2),
                "max": round(col_data.max(), 2),
            })
        profile[col] = info
    return profile

def detect_anomalies(df):
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty or len(numeric_df.columns) == 0:
        return [], 0
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(numeric_df)
    anomaly_indices = list(numeric_df[preds == -1].index)
    return anomaly_indices, len(anomaly_indices)

def calculate_health_score(profile, anomaly_count, total_rows):
    score = 100
    for col, info in profile.items():
        if info["null_pct"] > 50:
            score -= 15
        elif info["null_pct"] > 20:
            score -= 8
        elif info["null_pct"] > 5:
            score -= 3
    anomaly_pct = (anomaly_count / total_rows * 100) if total_rows > 0 else 0
    if anomaly_pct > 10:
        score -= 20
    elif anomaly_pct > 5:
        score -= 10
    return max(0, min(100, score))

def get_issues(profile, anomaly_count, total_rows):
    issues = []
    for col, info in profile.items():
        if info["null_pct"] > 20:
            issues.append({
                "severity": "🔴 Critical",
                "column": col,
                "issue": f"{info['null_pct']}% null values ({info['nulls']} rows)"
            })
        elif info["null_pct"] > 5:
            issues.append({
                "severity": "🟡 Warning",
                "column": col,
                "issue": f"{info['null_pct']}% null values ({info['nulls']} rows)"
            })
        if info["unique_pct"] == 100 and info["dtype"] != "object":
            issues.append({
                "severity": "🟢 Info",
                "column": col,
                "issue": "All values are unique — possible ID column"
            })
    anomaly_pct = round((anomaly_count / total_rows * 100), 2) if total_rows > 0 else 0
    if anomaly_pct > 5:
        issues.append({
            "severity": "🔴 Critical",
            "column": "Dataset-wide",
            "issue": f"{anomaly_count} anomalous rows detected ({anomaly_pct}%)"
        })
    elif anomaly_pct > 0:
        issues.append({
            "severity": "🟡 Warning",
            "column": "Dataset-wide",
            "issue": f"{anomaly_count} anomalous rows detected ({anomaly_pct}%)"
        })
    return issues