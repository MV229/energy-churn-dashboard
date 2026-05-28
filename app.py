import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Norwegian Energy Churn Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem; }
    div[data-testid="metric-container"] {
        background: #f0f4ff;
        border: 1px solid #dde3f4;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GENERATE SAMPLE DATA (if no CSV exists)
# Mirrors BCG simulation dataset structure
# ─────────────────────────────────────────────
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 14000

    tenure = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    consumption = np.random.normal(800, 300, n).clip(100, 5000)
    price_sensitivity = np.random.uniform(0, 1, n)
    contract_type = np.random.choice(["monthly", "annual", "2-year"], n, p=[0.4, 0.4, 0.2])
    industry = np.random.choice(
        ["Oil & Gas", "Retail", "Manufacturing", "Tech", "Construction"], n,
        p=[0.3, 0.2, 0.2, 0.15, 0.15]
    )
    num_products = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1])
    margin = np.random.normal(10, 5, n).clip(0, 30)

    # Churn logic: tenure + consumption + contract type drive churn (not price)
    contract_enc = {"monthly": 0.4, "annual": 0.15, "2-year": 0.05}
    churn_prob = (
        0.5 * np.exp(-tenure / 18)
        + 0.2 * (consumption < 400).astype(float)
        + np.array([contract_enc[c] for c in contract_type])
        + 0.05 * price_sensitivity
        - 0.03 * num_products
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    churn = (np.random.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "avg_consumption_kwh": consumption.round(1),
        "price_sensitivity": price_sensitivity.round(3),
        "contract_type": contract_type,
        "industry": industry,
        "num_products": num_products,
        "margin_pct": margin.round(2),
        "churn": churn
    })
    return df


@st.cache_resource
def train_model(df):
    le_contract = LabelEncoder()
    le_industry = LabelEncoder()
    df2 = df.copy()
    df2["contract_enc"] = le_contract.fit_transform(df2["contract_type"])
    df2["industry_enc"] = le_industry.fit_transform(df2["industry"])

    features = ["tenure_months", "avg_consumption_kwh", "price_sensitivity",
                "contract_enc", "industry_enc", "num_products", "margin_pct"]
    X = df2[features]
    y = df2["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, le_contract, le_industry, features, round(auc, 3)


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
DATA_PATH = "data/churn_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = generate_sample_data()

model, le_contract, le_industry, FEATURES, auc_score = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("⚡ Energy Churn")
    st.caption("Norwegian SME Market · BCG X Simulation")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Churn Drivers", "🗂 Customer Segments",
         "🤖 Risk Predictor", "📋 Data Explorer"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(f"Dataset: {len(df):,} customers")
    st.caption(f"Model AUC: {auc_score}")
    st.caption("Built by Vishakha Ratolikar")

# ─────────────────────────────────────────────
# FEATURE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Norwegian Energy Market — Churn Overview")
    st.caption("BCG X simulation | 14,000+ SME energy customers | Stavanger, Norway")

    total = len(df)
    churned = df["churn"].sum()
    churn_rate = df["churn"].mean()
    retained = total - churned
    avg_tenure = df[df["churn"] == 0]["tenure_months"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}", delta=f"-{churned:,}", delta_color="inverse")
    c3.metric("Churn Rate", f"{churn_rate:.1%}")
    c4.metric("Retained", f"{retained:,}")
    c5.metric("Avg Tenure (retained)", f"{avg_tenure:.0f} mo")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn distribution")
        fig = px.pie(
            values=[retained, churned],
            names=["Retained", "Churned"],
            color_discrete_sequence=["#1D9E75", "#D85A30"],
            hole=0.45
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn rate by contract type")
        ct = df.groupby("contract_type")["churn"].mean().reset_index()
        ct.columns = ["Contract Type", "Churn Rate"]
        ct["Churn Rate %"] = (ct["Churn Rate"] * 100).round(1)
        fig2 = px.bar(
            ct, x="Contract Type", y="Churn Rate %",
            color="Churn Rate %",
            color_continuous_scale=["#1D9E75", "#FAC775", "#D85A30"],
            text="Churn Rate %"
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Churn over contract tenure")
    tenure_churn = df.groupby("tenure_months")["churn"].mean().reset_index()
    fig3 = px.area(
        tenure_churn, x="tenure_months", y="churn",
        labels={"tenure_months": "Contract tenure (months)", "churn": "Churn rate"},
        color_discrete_sequence=["#378ADD"]
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# FEATURE 2 — CHURN DRIVERS
# ─────────────────────────────────────────────
elif page == "🔍 Churn Drivers":
    st.title("🔍 What Actually Drives Churn?")
    st.info(
        "**Key BCG Finding:** Price sensitivity ranked 5th in importance. "
        "Contract tenure and consumption patterns are the real drivers.",
        icon="💡"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature importance (Random Forest)")
        importances = pd.Series(
            model.feature_importances_,
            index=["Tenure", "Consumption", "Price sensitivity",
                   "Contract type", "Industry", "Num products", "Margin"]
        ).sort_values(ascending=True)

        colors = ["#D85A30" if i in ["Price sensitivity"] else "#378ADD"
                  for i in importances.index]

        fig = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in importances.values],
            textposition="outside"
        ))
        fig.update_layout(
            xaxis_title="Importance score",
            margin=dict(t=10, b=10),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟠 Price sensitivity — lower than expected. 🔵 Tenure + consumption — top predictors.")

    with col2:
        st.subheader("Churn rate vs consumption")
        bins = pd.cut(df["avg_consumption_kwh"], bins=10)
        cons_churn = df.groupby(bins, observed=True)["churn"].mean().reset_index()
        cons_churn["consumption_mid"] = cons_churn["avg_consumption_kwh"].apply(
            lambda x: round((x.left + x.right) / 2, 0)
        )
        fig2 = px.bar(
            cons_churn, x="consumption_mid", y="churn",
            labels={"consumption_mid": "Avg kWh/month", "churn": "Churn rate"},
            color="churn",
            color_continuous_scale=["#1D9E75", "#D85A30"]
        )
        fig2.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price sensitivity vs churn — scatter")
    sample = df.sample(min(2000, len(df)), random_state=1)
    fig3 = px.scatter(
        sample, x="price_sensitivity", y="tenure_months",
        color=sample["churn"].map({0: "Retained", 1: "Churned"}),
        color_discrete_map={"Retained": "#1D9E75", "Churned": "#D85A30"},
        opacity=0.5,
        labels={"price_sensitivity": "Price sensitivity score", "tenure_months": "Tenure (months)"},
        title="Scatter: tenure vs price sensitivity (2,000 sample)"
    )
    fig3.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# FEATURE 3 — CUSTOMER SEGMENTS
# ─────────────────────────────────────────────
elif page == "🗂 Customer Segments":
    st.title("🗂 Customer Segments")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn rate by industry")
        ind = df.groupby("industry")["churn"].agg(["mean", "count"]).reset_index()
        ind.columns = ["Industry", "Churn Rate", "Count"]
        ind["Churn Rate %"] = (ind["Churn Rate"] * 100).round(1)
        fig = px.bar(
            ind.sort_values("Churn Rate %", ascending=True),
            x="Churn Rate %", y="Industry", orientation="h",
            color="Churn Rate %",
            color_continuous_scale=["#1D9E75", "#FAC775", "#D85A30"],
            text="Churn Rate %"
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Number of products vs churn")
        np_churn = df.groupby("num_products")["churn"].mean().reset_index()
        np_churn.columns = ["Num Products", "Churn Rate"]
        np_churn["Churn Rate %"] = (np_churn["Churn Rate"] * 100).round(1)
        fig2 = px.line(
            np_churn, x="Num Products", y="Churn Rate %",
            markers=True, color_discrete_sequence=["#D85A30"]
        )
        fig2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("High-risk segment: short tenure + low consumption")
    high_risk = df[(df["tenure_months"] < 12) & (df["avg_consumption_kwh"] < 400)]
    low_risk = df[(df["tenure_months"] >= 24) & (df["avg_consumption_kwh"] >= 600)]

    r1, r2, r3 = st.columns(3)
    r1.metric("High-risk customers", f"{len(high_risk):,}")
    r2.metric("High-risk churn rate", f"{high_risk['churn'].mean():.1%}")
    r3.metric("Low-risk churn rate", f"{low_risk['churn'].mean():.1%}")

    st.caption("High-risk = tenure < 12 months AND consumption < 400 kWh/month")


# ─────────────────────────────────────────────
# FEATURE 4 — RISK PREDICTOR
# ─────────────────────────────────────────────
elif page == "🤖 Risk Predictor":
    st.title("🤖 Single Customer Churn Risk Predictor")
    st.caption("Enter customer details below to get an instant churn probability.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Contract tenure (months)", 1, 72, 12)
            consumption = st.number_input("Avg monthly usage (kWh)", 100, 5000, 800, step=50)

        with col2:
            price_sens = st.slider("Price sensitivity (0–1)", 0.0, 1.0, 0.5, step=0.05)
            contract = st.selectbox("Contract type", ["monthly", "annual", "2-year"])

        with col3:
            industry = st.selectbox("Industry", ["Oil & Gas", "Retail", "Manufacturing", "Tech", "Construction"])
            num_prod = st.selectbox("Number of products", [1, 2, 3, 4])
            margin = st.slider("Margin %", 0.0, 30.0, 10.0, step=0.5)

        submitted = st.form_submit_button("🔮 Predict churn risk", use_container_width=True)

    if submitted:
        contract_enc = le_contract.transform([contract])[0]
        industry_enc = le_industry.transform([industry])[0]

        X_input = pd.DataFrame([[
            tenure, consumption, price_sens,
            contract_enc, industry_enc, num_prod, margin
        ]], columns=FEATURES)

        prob = model.predict_proba(X_input)[0][1]
        risk_label = "🔴 High Risk" if prob > 0.6 else ("🟡 Medium Risk" if prob > 0.3 else "🟢 Low Risk")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{prob:.1%}")
        c2.metric("Risk level", risk_label)
        c3.metric("Retention probability", f"{1 - prob:.1%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#D85A30" if prob > 0.6 else ("#FAC775" if prob > 0.3 else "#1D9E75")},
                "steps": [
                    {"range": [0, 30], "color": "#e8f8f1"},
                    {"range": [30, 60], "color": "#fef9ec"},
                    {"range": [60, 100], "color": "#fdf0ec"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 60}
            },
            title={"text": "Churn Risk Score"}
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if prob > 0.5:
            st.warning(
                f"**Recommended action:** Offer a contract upgrade from *{contract}* to *annual* "
                f"and increase touchpoints in the first 6 months. Tenure is the #1 driver.",
                icon="⚠️"
            )
        else:
            st.success("This customer has a low churn risk. Standard retention program applies.", icon="✅")


# ─────────────────────────────────────────────
# FEATURE 5 — DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "📋 Data Explorer":
    st.title("📋 Raw Data Explorer")

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        industry_filter = st.multiselect(
            "Industry", df["industry"].unique().tolist(),
            default=df["industry"].unique().tolist()
        )
    with col2:
        contract_filter = st.multiselect(
            "Contract type", df["contract_type"].unique().tolist(),
            default=df["contract_type"].unique().tolist()
        )
    with col3:
        churn_filter = st.selectbox("Churn status", ["All", "Churned only", "Retained only"])

    filtered = df[
        df["industry"].isin(industry_filter) &
        df["contract_type"].isin(contract_filter)
    ]
    if churn_filter == "Churned only":
        filtered = filtered[filtered["churn"] == 1]
    elif churn_filter == "Retained only":
        filtered = filtered[filtered["churn"] == 0]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} customers")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered churn rate", f"{filtered['churn'].mean():.1%}")
    col2.metric("Avg tenure", f"{filtered['tenure_months'].mean():.0f} months")
    col3.metric("Avg consumption", f"{filtered['avg_consumption_kwh'].mean():.0f} kWh")

    st.dataframe(
        filtered.head(500).style.background_gradient(
            subset=["tenure_months", "avg_consumption_kwh"],
            cmap="RdYlGn"
        ),
        use_container_width=True,
        height=400
    )

    st.download_button(
        "⬇ Download filtered data as CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_churn_data.csv",
        mime="text/csv"
    )

