"""
DataWhisperer Pro – Production Hardened & Silent-Warning Edition
Fixes KeyError:0 (no categorical cols) and silences date-parsing warnings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# Logging & warnings
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", message="Could not infer format.*")  # silence dateutil

# ------------------------------------------------------------------
# Gemini setup
# ------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    logging.warning(f"Gemini unavailable: {e}")


# ------------------------------------------------------------------
# Safe Gemini wrapper
# ------------------------------------------------------------------
def safe_gemini(prompt: str, fallback: str = "AI service unavailable", max_tokens: int = 200) -> str:
    if not GEMINI_AVAILABLE:
        st.error("Gemini API key missing or invalid.  Check .env file.")
        return fallback
    if not st.session_state.get("allow_ai_sharing", True):
        return "AI response suppressed (privacy mode)."
    try:
        return (
            GEMINI_MODEL.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
            .text.strip()
        )
    except Exception as e:
        st.exception(e)
        return fallback


# ------------------------------------------------------------------
# Data loading & schema coercion
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    raw = file.read()
    encoding = "utf-8"
    try:
        raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "ISO-8859-1"
    file.seek(0)
    df = pd.read_csv(file, encoding=encoding)

    for col in df.columns:
        # datetime
        if df[col].dtype == "object":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
                continue
        # numeric
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            pass
        # categorical
        if df[col].dtype == "object" and df[col].nunique() < 0.3 * len(df):
            df[col] = df[col].astype("category")
    return df


# ------------------------------------------------------------------
# Session helpers
# ------------------------------------------------------------------
def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_comprehensive_eda(df: pd.DataFrame) -> Dict[str, go.Figure]:
    return create_comprehensive_eda(df)


# ------------------------------------------------------------------
# Color-blind friendly palette
# ------------------------------------------------------------------
COLOR_PALETTE = px.colors.qualitative.Safe


# ------------------------------------------------------------------
# EDA figures
# ------------------------------------------------------------------
def create_comprehensive_eda(df: pd.DataFrame) -> Dict[str, go.Figure]:
    figures = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # 1. Correlation
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        mask = pd.DataFrame(
            np.triu(np.ones_like(corr, dtype=bool), k=1), index=corr.index, columns=corr.columns
        )
        fig = go.Figure(
            go.Heatmap(
                z=corr.mask(mask),
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
                zmid=0,
                text=np.round(corr, 2),
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig.update_layout(title="🔥 Correlation Matrix", height=500)
        figures["correlation"] = fig

    # 2. Distribution histograms
    max_hist = min(len(num_cols), 6)
    if max_hist:
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=[f"{c}" for c in num_cols[:max_hist]]
        )
        for i, col in enumerate(num_cols[:max_hist]):
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    showlegend=False,
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                ),
                row=(i // 3) + 1,
                col=(i % 3) + 1,
            )
        fig.update_layout(title="📊 Distributions", height=600)
        figures["distributions"] = fig

    # 3. Box-plots
    max_box = min(len(num_cols), 8)
    if max_box:
        fig = go.Figure()
        for i, col in enumerate(num_cols[:max_box]):
            fig.add_trace(
                go.Box(
                    y=df[col],
                    name=f"{col}",
                    boxpoints="outliers",
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                )
            )
        fig.update_layout(title="📦 Anomaly Detection", height=400)
        figures["boxplots"] = fig

    # 4. Scatter matrix
    if len(num_cols) >= 2:
        cols = num_cols[: min(4, len(num_cols))]
        fig = px.scatter_matrix(
            df[cols],
            dimensions=cols,
            title="🎯 Multi-Dimensional Analysis",
            height=700,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.7))
        figures["scatter_matrix"] = fig

    # 5. PCA + clustering
    if len(num_cols) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[num_cols].dropna())
        pca = PCA(n_components=2).fit(scaled)
        comps = pca.transform(scaled)
        n_clusters = min(5, max(2, len(df) // 500))
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(
            comps
        )
        fig = px.scatter(
            x=comps[:, 0],
            y=comps[:, 1],
            color=clusters.astype(str),
            title="🧬 AI Pattern Recognition",
            labels={
                "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            },
            height=500,
        )
        figures["pca"] = fig

    # 6. Trends
    if len(df) > 20 and len(num_cols) > 0:
        max_window = min(50, max(5, len(df) // 50))
        fig = go.Figure()
        for col in num_cols[:3]:
            ma = df[col].rolling(window=max_window).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    opacity=0.4,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma,
                    mode="lines",
                    name=f"{col} trend",
                    line=dict(width=3),
                )
            )
        fig.update_layout(title="📈 Trend Analysis", height=400, hovermode="x unified")
        figures["trends"] = fig

    # 7. Categorical bar (only if any)
    if len(cat_cols) > 0:
        col = cat_cols[0]
        if df[col].nunique() <= 20:
            counts = df[col].value_counts().head(10)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title=f"🏷️ {col}",
                labels={"x": col, "y": "Count"},
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig.update_layout(height=400)
            figures["categorical"] = fig

    # 8. 3-D scatter
    if len(num_cols) >= 3:
        fig = px.scatter_3d(
            df,
            x=num_cols[0],
            y=num_cols[1],
            z=num_cols[2],
            color=num_cols[0],
            title="🌐 3D Data Universe",
            height=600,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        figures["3d"] = fig

    return figures


# ------------------------------------------------------------------
# AutoML (leak-proof)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def quick_ml(df: pd.DataFrame, target: str) -> Optional[Dict[str, Any]]:
    if target not in df.columns:
        return None

    y = df[target]
    X = df.drop(columns=[target])

    num_features = X.select_dtypes(include=[np.number])
    cat_features = X.select_dtypes(exclude=[np.number])

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_features.columns),
            ("cat", categorical_pipe, cat_features.columns),
        ]
    )

    is_classification = y.dtype == "object" or y.nunique() < 10
    if is_classification:
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        task = "Classification"
    else:
        y_enc = y
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        task = "Regression"

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc if is_classification else None,
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)

    importances = (
        pd.DataFrame(
            {
                "feature": pipe.named_steps["prep"].get_feature_names_out().tolist(),
                "importance": pipe.named_steps["model"].feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    insights = []
    if score > 0.9:
        insights.append("🏆 Exceptional model performance!")
    elif score > 0.75:
        insights.append("✅ Strong predictive capability")
    top = importances.iloc[0]
    insights.append(f"🎯 {top['feature']} drives prediction ({top['importance']*100:.1f}%)")
    tip = safe_gemini(
        f"In 10 words, business action from {score:.1%} {task.lower()} accuracy on {target}?",
        "Prioritize high-value opportunities",
    )
    insights.append(f"💡 {tip}")
    return dict(score=score, task=task, features=importances, insights=insights)


# ------------------------------------------------------------------
# AI helpers
# ------------------------------------------------------------------
def generate_ai_insights(df: pd.DataFrame) -> List[str]:
    insights = []
    num = df.select_dtypes(include=[np.number])
    for col in num.columns[:3]:
        cv = num[col].std() / (num[col].mean() or 1)
        if cv > 0.5:
            insights.append(f"🎯 High volatility in {col} (CV={cv:.2f})")
    if len(num.columns) > 1:
        corr = num.corr()
        mask = (corr.abs() > 0.7) & (corr.abs() < 1)
        pairs = [(corr.index[i], corr.columns[j]) for i, j in zip(*np.where(mask))]
        for a, b in pairs[:3]:
            insights.append(f"🔗 Strong correlation: {a} ↔ {b}")
    quality = 100 * (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
    insights.append(f"💎 Data quality: {quality:.1f}%")
    recs = safe_gemini(
        f"Suggest 3 specific analyses for {len(num.columns)} numeric cols. Return one bullet per line.",
        "Correlation\nDistribution\nOutliers",
    ).splitlines()[:3]
    if recs:
        insights.append(f"🤖 AI suggests: {recs[0]}")
    return insights[:5]


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.set_page_config("DataWhisperer Pro – Hardened", "🎯", layout="wide")

if "df" not in st.session_state:
    st.session_state.df = None
if "allow_ai_sharing" not in st.session_state:
    st.session_state.allow_ai_sharing = True

with st.sidebar:
    st.header("📁 Data Control Center")
    st.checkbox("Allow AI to see sample rows", value=True, key="allow_ai_sharing")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        file_hash = _file_hash(uploaded_file.getbuffer())
        if st.session_state.get("file_hash") != file_hash:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.file_hash = file_hash

        df = st.session_state.df
        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))

        st.subheader("🤖 AI Insights")
        for insight in generate_ai_insights(df):
            st.info(insight)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Smart EDA", "📈 Custom Analysis", "🤖 AutoML", "🧪 AI Lab"])

if st.session_state.df is not None:
    df = st.session_state.df

    with tab1:
        st.header("📊 Intelligent EDA Dashboard")
        with st.spinner("Analyzing…"):
            figures = run_comprehensive_eda(df)
        st.subheader("💡 Executive Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            miss = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            st.metric("Quality", f"{100-miss:.1f}%")
        with c2:
            st.metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
        with c3:
            st.metric("Categorical", len(df.select_dtypes(exclude=[np.number]).columns))
        with c4:
            corr = df.select_dtypes(include=[np.number]).corr()
            high = (corr.abs() > 0.7).sum().sum() - len(corr)
            st.metric("Correlations", high // 2)
        for key, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📈 Interactive Visualization Studio")
        c1, c2 = st.columns([1, 3])
        with c1:
            viz = st.selectbox(
                "Viz",
                ["Scatter", "Histogram", "Box", "Violin", "3D Scatter", "Bubble", "Heatmap"],
            )
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            x_col, y_col, z_col, size_col, color_col = None, None, None, None, None
            if viz in ["Histogram", "Box", "Violin"]:
                x_col = st.selectbox("Column", num_cols)
            elif viz in ["Scatter", "Bubble"]:
                x_col = st.selectbox("X", num_cols)
                y_col = st.selectbox("Y", num_cols, index=1 if len(num_cols) > 1 else 0)
                if viz == "Bubble" and len(num_cols) > 2:
                    size_col = st.selectbox("Size", num_cols, index=2)
            elif viz == "3D Scatter" and len(num_cols) >= 3:
                x_col = st.selectbox("X", num_cols)
                y_col = st.selectbox("Y", num_cols, index=1)
                z_col = st.selectbox("Z", num_cols, index=2)
            color_col = st.selectbox("Color", ["None"] + list(df.columns))
            if color_col == "None":
                color_col = None
        with c2:
            fig = None
            if viz == "Scatter" and x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            elif viz == "Histogram" and x_col:
                fig = px.histogram(df, x=x_col, marginal="rug", color=color_col)
            elif viz == "Box" and x_col:
                fig = px.box(df, y=x_col, color=color_col)
            elif viz == "Violin" and x_col:
                fig = px.violin(df, y=x_col, box=True, color=color_col)
            elif viz == "3D Scatter" and z_col:
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col)
            elif viz == "Bubble" and size_col:
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, size_max=60)
            elif viz == "Heatmap" and len(num_cols) > 1:
                fig = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale="Viridis")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🤖 Automated Machine Learning")
        col1, col2 = st.columns([1, 2])
        with col1:
            target = st.selectbox("🎯 Target", df.columns)
            if st.button("🚀 Launch AutoML", type="primary"):
                with st.spinner("Training…"):
                    results = quick_ml(df, target)
                    if results:
                        st.success("✅ Model Ready!")
                        st.metric("Score", f"{results['score']:.3f}")
                        st.caption(results["task"])
                        for insight in results["insights"]:
                            st.info(insight)
        with col2:
            if "results" in locals() and results:
                fig = px.bar(
                    results["features"],
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("🧪 AI Laboratory")
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_area("Ask AI about your data:", height=100)
            if st.button("🤖 Ask AI") and query:
                sample = (
                    df.head(3).to_dict(orient="records")
                    if st.session_state.allow_ai_sharing
                    else "[redacted]"
                )
                prompt = f"""
                Dataset: {df.shape}
                Sample keys: {list(sample[0].keys()) if sample != '[redacted]' else sample}
                User: {query}
                Answer (≤80 words):
                """
                with st.spinner("Querying…"):
                    answer = safe_gemini(prompt)
                st.success("AI Response:")
                st.write(answer)

        with col2:
            st.subheader("🔬 Anomaly Detection")
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols):
                col = st.selectbox("Column", num_cols)
                if st.button("🔍 Detect"):
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                    anomalies = df[mask]
                    st.info(f"Found {len(anomalies)} anomalies")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df[col], mode="markers", name="Normal"))
                    fig.add_trace(
                        go.Scatter(
                            y=anomalies[col], x=anomalies.index, mode="markers", name="Anomalies"
                        )
                    )
                    fig.update_layout(title=f"Anomalies in {col}")
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown(
        """
    ## 🚀 Welcome to DataWhisperer Pro – Hardened Edition
    Upload a CSV on the left to unlock AI-powered insights!
    """
    )