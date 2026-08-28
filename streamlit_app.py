"""
AI-Powered Vulnerability Management Dashboard
----------------------------------------------
Run with: streamlit run streamlit_app.py

Requires processed_cves.csv and exploit_model.json to already exist -
these are created by running vuln_management_prototype.py first.

Install dependencies:
    pip install streamlit pandas xgboost shap plotly --break-system-packages
"""

import json
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="Vulnerability Risk Dashboard", layout="wide")

FEATURE_COLS = [
    "base_score",
    "age_days",
    "vendor_frequency",
    "no_privileges_required",
    "no_user_interaction",
    "network_attack_vector",
]

FEATURE_LABELS = {
    "base_score": "CVSS base score",
    "age_days": "Vulnerability age (days)",
    "vendor_frequency": "Vendor appears often in dataset",
    "no_privileges_required": "No privileges required to attack",
    "no_user_interaction": "No user interaction required",
    "network_attack_vector": "Exploitable remotely over network",
}


# ---------------------------------------------------------------------------
# Data + model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_cve_data():
    try:
        df = pd.read_csv("processed_cves.csv")
        # CSV round-tripping can turn numeric columns into text (object dtype).
        # Force them back to numeric so XGBoost/SHAP accept them.
        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model("exploit_model.json")
        return model
    except Exception:
        return None


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_real_scan():
    """Loads the real laptop scan if scan_real_assets.py has been run."""
    try:
        df = pd.read_csv("real_assets_scored.csv")
        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        df["internet_facing"] = df["internet_facing"].astype(bool)
        df["sensitive_data"] = df["sensitive_data"].astype(bool)
        return df
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Sample asset inventory - a fictional company's systems
# Used as a fallback only if no real scan (real_assets_scored.csv) is found.
# ---------------------------------------------------------------------------

SAMPLE_ASSETS = pd.DataFrame([
    {"asset_name": "Payment API gateway", "vendor_keyword": "apache", "internet_facing": True, "sensitive_data": True},
    {"asset_name": "Customer login service", "vendor_keyword": "microsoft", "internet_facing": True, "sensitive_data": True},
    {"asset_name": "Internal HR portal", "vendor_keyword": "linux", "internet_facing": False, "sensitive_data": True},
    {"asset_name": "Marketing website", "vendor_keyword": "wordpress", "internet_facing": True, "sensitive_data": False},
    {"asset_name": "Backup file server", "vendor_keyword": "openssh", "internet_facing": False, "sensitive_data": True},
    {"asset_name": "Analytics dashboard", "vendor_keyword": "nodejs", "internet_facing": True, "sensitive_data": False},
    {"asset_name": "Print server", "vendor_keyword": "windows", "internet_facing": False, "sensitive_data": False},
])


def match_assets_to_cves(cve_df, assets_df):
    """
    Matches each asset to CVEs whose vendor field contains the asset's
    vendor keyword. This is a simple substring match for the prototype -
    a real system would match against precise product/version strings.
    """
    matches = []
    for _, asset in assets_df.iterrows():
        keyword = asset["vendor_keyword"].lower()
        matched = cve_df[cve_df["vendor"].str.lower().str.contains(keyword, na=False)]
        for _, cve in matched.iterrows():
            matches.append({**asset.to_dict(), **cve.to_dict()})
    return pd.DataFrame(matches)


def compute_priority_score(row, model):
    """
    Combines the model's predicted exploit probability with simple
    business-context weighting (exposure + data sensitivity).
    This weighted-sum approach is intentionally simple for a prototype -
    document this as a limitation/future-work item in your report.
    """
    features = row[FEATURE_COLS].astype(float).values.reshape(1, -1)
    exploit_prob = model.predict_proba(features)[0][1]

    context_weight = 1.0
    if row["internet_facing"]:
        context_weight += 0.3
    if row["sensitive_data"]:
        context_weight += 0.3

    priority = min(exploit_prob * context_weight, 1.0)
    return exploit_prob, priority


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("AI-powered vulnerability management dashboard")

cve_df = load_cve_data()
model = load_model()
real_scan_df = load_real_scan()

if cve_df is None or model is None:
    st.error(
        "Could not find processed_cves.csv or exploit_model.json. "
        "Run vuln_management_prototype.py first to generate these files, "
        "then place them in the same folder as this dashboard script."
    )
    st.stop()

if real_scan_df is not None:
    st.caption("Live scan of software installed on this machine")
    matched_df = real_scan_df
else:
    st.caption("Explainable exploit-risk prioritization for a sample organization "
               "(run scan_real_assets.py to use your own machine's real data instead)")
    matched_df = match_assets_to_cves(cve_df, SAMPLE_ASSETS)

if matched_df.empty:
    st.warning(
        "No CVEs matched the sample asset inventory's vendor keywords. "
        "This can happen with a small dataset - try increasing total_results "
        "in the data collection step, or edit SAMPLE_ASSETS keywords."
    )
    st.stop()

# Compute scores for every matched asset-CVE pair
exploit_probs, priorities = [], []
for _, row in matched_df.iterrows():
    prob, priority = compute_priority_score(row, model)
    exploit_probs.append(prob)
    priorities.append(priority)

matched_df["exploit_probability"] = exploit_probs
matched_df["priority_score"] = priorities
matched_df = matched_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

# --- Summary metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Assets scanned", len(SAMPLE_ASSETS))
col2.metric("CVEs matched", len(matched_df))
col3.metric("Critical priority (>0.7)", int((matched_df["priority_score"] > 0.7).sum()))

st.divider()

# --- Ranked table ---
st.subheader("Ranked vulnerabilities")
# The real scan has 'installed_version' instead of 'vendor' (the fictional
# sample data source) - show whichever column is actually present.
extra_col = "installed_version" if "installed_version" in matched_df.columns else "vendor"
display_df = matched_df[["asset_name", "cve_id", extra_col, "base_score", "priority_score"]].copy()
display_df["priority_score"] = display_df["priority_score"].round(2)
display_df["base_score"] = display_df["base_score"].round(1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# --- SHAP explanation for a selected CVE ---
st.subheader("Explain a vulnerability's risk score")
selected_cve = st.selectbox("Select a CVE to see why it was scored this way", matched_df["cve_id"].unique())

selected_row = matched_df[matched_df["cve_id"] == selected_cve].iloc[0]
features_df = pd.DataFrame([selected_row[FEATURE_COLS].astype(float)], columns=FEATURE_COLS)

explainer = get_explainer(model)
shap_values = explainer.shap_values(features_df)[0]

labels = [FEATURE_LABELS[f] for f in FEATURE_COLS]
colors = ["#D85A30" if v > 0 else "#1D9E75" for v in shap_values]

fig = go.Figure(go.Bar(
    x=shap_values,
    y=labels,
    orientation="h",
    marker_color=colors,
))
fig.update_layout(
    xaxis_title="Contribution to risk prediction",
    height=350,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"{selected_row['asset_name']} runs software affected by {selected_cve} "
    f"(CVSS base score: {selected_row['base_score']:.1f}). "
    f"Predicted exploit probability: {selected_row['exploit_probability']:.0%}. "
    f"Red bars increased the risk score, green bars decreased it."
)