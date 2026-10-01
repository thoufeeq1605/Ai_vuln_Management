# Run with: streamlit run streamlit_app.py
# Needs processed_cves.csv and exploit_model.json (from vuln_management_prototype.py)

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


@st.cache_data
def load_cve_data():
    try:
        df = pd.read_csv("processed_cves.csv")
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
    matches = []
    for _, asset in assets_df.iterrows():
        keyword = asset["vendor_keyword"].lower()
        matched = cve_df[cve_df["vendor"].str.lower().str.contains(keyword, na=False)]
        for _, cve in matched.iterrows():
            matches.append({**asset.to_dict(), **cve.to_dict()})
    return pd.DataFrame(matches)


def compute_priority_score(row, model):
    features = row[FEATURE_COLS].astype(float).values.reshape(1, -1)
    exploit_prob = model.predict_proba(features)[0][1]

    context_score = 0.0
    if row["internet_facing"]:
        context_score += 0.5
    if row["sensitive_data"]:
        context_score += 0.5

    priority = (0.7 * exploit_prob) + (0.3 * context_score)
    return exploit_prob, priority


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
    data_source = st.radio(
        "Data source",
        ["Real laptop scan", "Sample organization (fictional)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if data_source == "Real laptop scan":
        st.caption("Live scan of software installed on the developer's own laptop")
        matched_df = real_scan_df
    else:
        st.caption("Explainable exploit-risk prioritization for a fictional sample organization")
        matched_df = match_assets_to_cves(cve_df, SAMPLE_ASSETS)
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

exploit_probs, priorities = [], []
for _, row in matched_df.iterrows():
    prob, priority = compute_priority_score(row, model)
    exploit_probs.append(prob)
    priorities.append(priority)

matched_df["exploit_probability"] = exploit_probs
matched_df["priority_score"] = priorities
matched_df = matched_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Assets scanned", len(SAMPLE_ASSETS))
col2.metric("CVEs matched", len(matched_df))
col3.metric("Critical priority (>0.7)", int((matched_df["priority_score"] > 0.7).sum()))

st.divider()

st.subheader("Ranked vulnerabilities")
extra_col = "installed_version" if "installed_version" in matched_df.columns else "vendor"
display_df = matched_df[["asset_name", "cve_id", extra_col, "base_score", "priority_score"]].copy()
display_df["priority_score"] = display_df["priority_score"].round(2)
display_df["base_score"] = display_df["base_score"].round(1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Explain a vulnerability's risk score")
selected_cve = st.selectbox("Select a CVE to see why it was scored this way", matched_df["cve_id"].unique())

selected_row = matched_df[matched_df["cve_id"] == selected_cve].iloc[0]
features_df = pd.DataFrame([selected_row[FEATURE_COLS].astype(float)], columns=FEATURE_COLS)

explainer = get_explainer(model)
shap_values = explainer.shap_values(features_df)[0]

sorted_pairs = sorted(zip(FEATURE_COLS, shap_values), key=lambda pair: abs(pair[1]))
sorted_features = [p[0] for p in sorted_pairs]
sorted_values = [p[1] for p in sorted_pairs]
sorted_labels = [FEATURE_LABELS[f] for f in sorted_features]

risk_color = "#E4572E"
safe_color = "#2E9E5B"
colors = [risk_color if v > 0 else safe_color for v in sorted_values]

fig = go.Figure(go.Bar(
    x=sorted_values,
    y=sorted_labels,
    orientation="h",
    marker=dict(color=colors, line=dict(width=0)),
    text=[f"{v:+.2f}" for v in sorted_values],
    textposition="outside",
    textfont=dict(size=13),
))
fig.add_vline(x=0, line_width=1.5, line_color="rgba(150,150,150,0.6)")
fig.update_layout(
    title=dict(text="What drove this risk score", font=dict(size=16)),
    xaxis_title="Contribution to predicted risk",
    height=380,
    margin=dict(l=10, r=40, t=50, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    showlegend=False,
    xaxis=dict(zeroline=False, gridcolor="rgba(150,150,150,0.15)"),
    yaxis=dict(automargin=True),
)
st.plotly_chart(fig, use_container_width=True)

increasing = [(FEATURE_LABELS[f], v) for f, v in zip(sorted_features, sorted_values) if v > 0]
decreasing = [(FEATURE_LABELS[f], v) for f, v in zip(sorted_features, sorted_values) if v < 0]
increasing.sort(key=lambda p: -p[1])
decreasing.sort(key=lambda p: p[1])
top_increasing = [label for label, _ in increasing[:2]]
top_decreasing = [label for label, _ in decreasing[:2]]

summary_parts = []
if top_increasing:
    summary_parts.append(f"flagged as **higher risk** mainly because of: {', '.join(top_increasing).lower()}")
if top_decreasing:
    summary_parts.append(f"risk was **reduced** by: {', '.join(top_decreasing).lower()}")
summary_text = " — ".join(summary_parts) if summary_parts else "no single factor stood out strongly."

st.info(
    f"**{selected_cve}** affecting **{selected_row['asset_name']}** was {summary_text}.\n\n"
    f"CVSS base score: {selected_row['base_score']:.1f} · "
    f"Predicted exploit probability: {selected_row['exploit_probability']:.0%} · "
    f"Final priority score: {selected_row['priority_score']:.2f}"
)
st.caption("Orange bars pushed the risk score up, green bars pulled it down - longer bars mean stronger influence.")
