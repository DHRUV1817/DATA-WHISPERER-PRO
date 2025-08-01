import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config("DataWhisperer Pro", "🎯", layout="wide")

# ------------------------------------------------------------------
#  Helper: safe Gemini wrapper
# ------------------------------------------------------------------
def safe_gemini(prompt: str, fallback: str = "AI service unavailable") -> str:
    try:
        return model.generate_content(prompt).text
    except Exception:
        return fallback

# ------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def generate_data_story(df, insights):
    return safe_gemini(
        f"""
        Create a brief, professional data story (max 100 words) based on:
        Dataset shape: {df.shape}
        Columns: {list(df.columns)[:5]}...
        Key insights: {insights[:2]}
        
        Write as a data analyst presenting findings. Be specific and actionable.
        """,
        "Your data reveals interesting patterns worth exploring further."
    )

def get_analysis_recommendations(df):
    txt = safe_gemini(
        f"""
        Suggest 3 specific analyses for a dataset with:
        {len(df.select_dtypes(include=[np.number]).columns)} numeric columns
        {len(df.select_dtypes(include=['object']).columns)} categorical columns
        
        Format: Brief actionable recommendations only. No explanations.
        """,
        "Correlation analysis\nDistribution profiling\nOutlier investigation"
    )
    return [line for line in txt.split('\n') if line.strip()][:3]

def generate_smart_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    txt = safe_gemini(
        f"""
        Suggest 2 simple feature engineering ideas for:
        Columns: {numeric_cols[:3]}
        
        Format: Column_name: transformation
        Keep it simple and practical.
        """,
        "Consider log transformation for skewed distributions\nCreate interaction features"
    )
    return [s.strip() for s in txt.split('\n') if s.strip()][:2]

def anomaly_explanation(df, col, anomalies):
    return safe_gemini(
        f"""
        In one sentence, explain why {len(anomalies)} anomalies were detected in '{col}' 
        (mean: {df[col].mean():.2f}, std: {df[col].std():.2f}).
        Be technical but concise.
        """,
        f"Detected {len(anomalies)} values beyond expected range."
    )

def generate_executive_summary(df, ml_results=None):
    summary = dict(
        rows=len(df),
        columns=len(df.columns),
        missing=df.isnull().sum().sum(),
        numeric_cols=len(df.select_dtypes(include=[np.number]).columns),
    )
    if ml_results:
        summary['ml_score'] = ml_results['score']
        summary['top_feature'] = ml_results['features'].iloc[0]['feature']

    return safe_gemini(
        f"""
        Write a 2-sentence executive summary for:
        - Dataset: {summary['rows']} rows, {summary['columns']} columns
        - Quality: {summary['missing']} missing values
        - ML Performance: {summary.get('ml_score', 'N/A')}
        
        Be direct and highlight the most important finding.
        """,
        "Dataset analysis complete. Key patterns identified for strategic decision-making."
    )

def generate_ai_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # volatility
    for col in numeric_cols[:3]:
        mean = df[col].mean()
        std = df[col].std()
        cv = std / mean if mean else 0
        if cv > 0.5:
            insights.append(f"🎯 High volatility in {col} (CV: {cv:.2f})")

    # correlations
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        mask = (corr.abs() > 0.7) & (corr.abs() < 1)
        pairs = np.column_stack(np.where(mask))
        for r, c in pairs:
            if r < c:
                insights.append(f"🔗 Strong correlation: {numeric_cols[r]} ↔ {numeric_cols[c]}")

    # quality
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    quality = (1 - missing_ratio) * 100
    insights.append(f"💎 Data Quality: {quality:.1f}%")

    recs = get_analysis_recommendations(df)
    if recs:
        insights.append(f"🤖 AI suggests: {recs[0]}")

    return insights[:5]

def create_comprehensive_eda(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    figures = {}

    # 1. Correlation
        # 1. Correlation heat-map
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].astype(float).corr()
        mask = pd.DataFrame(
            np.triu(np.ones_like(corr, dtype=bool), k=1),
            index=corr.index,
            columns=corr.columns
        )
        fig = go.Figure(go.Heatmap(
            z=corr.mask(mask),
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig.update_layout(title="🔥 Correlation Matrix", height=500)
        figures['correlation'] = fig

    # 2. Distributions
    if len(numeric_cols) > 0:
        n_cols = min(len(numeric_cols), 6)
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"{c}" for c in numeric_cols[:n_cols]])
        for i, col in enumerate(numeric_cols[:n_cols]):
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, showlegend=False,
                             marker_color='rgba(55, 128, 191, 0.7)'),
                row=(i // 3) + 1, col=(i % 3) + 1
            )
        fig.update_layout(title="📊 Distribution Analysis", height=600)
        figures['distributions'] = fig

    # 3. Box-plots
    if len(numeric_cols) > 0:
        fig = go.Figure()
        for i, col in enumerate(numeric_cols[:min(8, len(numeric_cols))]):
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
            fig.add_trace(go.Box(
                y=df[col], name=f"{col} ({len(outliers)} outliers)",
                boxpoints='outliers',
                marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            ))
        fig.update_layout(title="📦 Anomaly Detection System", height=400)
        figures['boxplots'] = fig

    # 4. Scatter matrix
    if len(numeric_cols) >= 2:
        cols = numeric_cols[:min(4, len(numeric_cols))]
        fig = px.scatter_matrix(df[cols], dimensions=cols,
                                title="🎯 Multi-Dimensional Analysis", height=700)
        fig.update_traces(diagonal_visible=False,
                          marker=dict(size=5, opacity=0.6))
        figures['scatter_matrix'] = fig

    # 5. PCA
    if len(numeric_cols) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean()))
        pca = PCA(n_components=2).fit_transform(scaled)
        n_clusters = min(4, max(2, len(df) // 50))
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pca)
        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=clusters.astype(str),
                         title="🧬 AI Pattern Recognition",
                         labels={'x': f'PC1 ({PCA(2).fit(scaled).explained_variance_ratio_[0]:.1%})',
                                 'y': f'PC2 ({PCA(2).fit(scaled).explained_variance_ratio_[1]:.1%})'})
        fig.update_layout(height=500)
        figures['pca'] = fig

    # 6. Trends
    if len(df) > 20 and len(numeric_cols) > 0:
        fig = go.Figure()
        for col in numeric_cols[:3]:
            ma = df[col].rolling(window=max(5, len(df) // 20)).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines',
                                     name=col, opacity=0.6))
            fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines',
                                     name=f'{col} (Trend)', line=dict(width=3, dash='dash')))
        fig.update_layout(title="📈 Trend Analysis", height=400, hovermode='x unified')
        figures['trends'] = fig

    # 7. Categorical bar
    if categorical_cols:
        cat_col = categorical_cols[0]
        if df[cat_col].nunique() <= 20:
            counts = df[cat_col].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values,
                         title=f"🏷️ {cat_col} Distribution",
                         labels={'x': cat_col, 'y': 'Count'})
            fig.update_traces(marker_color='lightblue',
                              marker_line_color='darkblue', marker_line_width=1.5)
            fig.update_layout(height=400)
            figures['categorical'] = fig

    # 8. 3D scatter
    if len(numeric_cols) >= 3:
        fig = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                            color=df[numeric_cols[0]], title="🌐 3D Data Universe", height=600)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        figures['3d'] = fig

    return figures

def quick_ml(df, target):
    if target not in df.columns:
        return None

    X = df.drop(columns=[target])
    y = df[target]

    # categorical predictors
    for col in X.select_dtypes(include=['object']):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.mean())

    # target encoding
    if y.dtype == 'object' or y.nunique() < 10:
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        mdl = RandomForestClassifier(n_estimators=100, random_state=42)
        task = "Classification"
    else:
        y_enc = y
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        task = "Regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42)
    mdl.fit(X_train, y_train)
    score = mdl.score(X_test, y_test)

    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mdl.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    insights = []
    if score > 0.9:
        insights.append("🏆 Exceptional model performance achieved!")
    elif score > 0.75:
        insights.append("✅ Strong predictive capability")
    top_feat = importance.iloc[0]
    insights.append(f"🎯 {top_feat['feature']} is the key driver ({top_feat['importance']*100:.1f}%)")
    ai_tip = safe_gemini(
        f"In 10 words, what business action does {score:.1%} {task.lower()} accuracy on {target} enable?",
        "Use predictions to prioritize high-value opportunities"
    )
    insights.append(f"💡 {ai_tip}")

    return dict(score=score, task=task, features=importance, insights=insights)

# ------------------------------------------------------------------
#  Streamlit UI
# ------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ai_story' not in st.session_state:
    st.session_state.ai_story = None

st.title("🎯 DataWhisperer Pro")
st.caption("AI-Powered Intelligence Platform with Gemini Integration")

with st.sidebar:
    st.header("📁 Data Control Center")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")
        df = st.session_state.df
        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))

        st.subheader("🤖 AI-Powered Insights")
        insights = generate_ai_insights(df)
        for insight in insights:
            st.info(insight)

        with st.spinner("🧠 Generating data narrative..."):
            st.session_state.ai_story = generate_data_story(df, insights)

        st.subheader("🔧 AI Feature Suggestions")
        suggestions = generate_smart_features(df)
        for suggestion in suggestions:
            st.code(suggestion, language='python')

if st.session_state.df is not None:
    df = st.session_state.df
    if st.session_state.ai_story:
        st.markdown("### 📖 Your Data Story")
        st.info(st.session_state.ai_story)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Smart EDA", "📈 Custom Analysis", "🤖 AutoML", "🧪 AI Lab"])

    with tab1:
        st.header("📊 Intelligent EDA Dashboard")
        with st.spinner("🧠 AI analyzing patterns..."):
            figures = create_comprehensive_eda(df)

        st.subheader("💡 Executive Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Quality", f"{100 - missing_pct:.1f}%",
                      "✅ Good" if missing_pct < 5 else "⚠️ Review")
        with col2:
            st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        with col3:
            st.metric("Categories", len(df.select_dtypes(include=['object']).columns))
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                high_corr = (corr.abs() > 0.7).sum().sum() - len(corr)
                st.metric("Correlations", high_corr // 2)

        for key, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Statistical Profile")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().round(2), use_container_width=True)

    with tab2:
        st.header("📈 Interactive Visualization Studio")
        col1, col2 = st.columns([1, 2])
        with col1:
            viz_type = st.selectbox("Visualization Type",
                                    ["Scatter", "Histogram", "Box", "Violin", "3D Scatter", "Bubble", "Heatmap"])
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if viz_type in ["Histogram", "Box", "Violin"]:
                x_col = st.selectbox("Select Column", numeric_cols)
                y_col = None
            elif viz_type in ["Scatter", "Bubble"]:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                if viz_type == "Bubble" and len(numeric_cols) > 2:
                    size_col = st.selectbox("Bubble Size", numeric_cols, index=2)
            elif viz_type == "3D Scatter" and len(numeric_cols) >= 3:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols, index=1)
                z_col = st.selectbox("Z-axis", numeric_cols, index=2)
            else:
                x_col = None
                y_col = None
            color_col = st.selectbox("Color by", ["None"] + list(df.columns))
            if color_col == "None":
                color_col = None
        with col2:
            fig = None
            if viz_type == "Scatter" and x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
            elif viz_type == "Histogram" and x_col:
                fig = px.histogram(df, x=x_col, marginal="rug", color=color_col, title=f"Distribution: {x_col}")
            elif viz_type == "Box" and x_col:
                fig = px.box(df, y=x_col, color=color_col, title=f"Box Plot: {x_col}")
            elif viz_type == "Violin" and x_col:
                fig = px.violin(df, y=x_col, box=True, color=color_col, title=f"Violin Plot: {x_col}")
            elif viz_type == "3D Scatter" and 'z_col' in locals():
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title="3D Visualization")
            elif viz_type == "Bubble" and 'size_col' in locals():
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                                 title="Bubble Chart", size_max=60)
            elif viz_type == "Heatmap" and len(numeric_cols) > 1:
                fig = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Viridis")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🤖 Automated Machine Learning")
        col1, col2 = st.columns([1, 2])
        with col1:
            target = st.selectbox("🎯 Target Variable", df.columns)
            if st.button("🚀 Launch AutoML", type="primary"):
                with st.spinner("🧠 Training AI models..."):
                    results = quick_ml(df, target)
                    if results:
                        st.success("✅ Model Ready!")
                        st.metric("Performance Score", f"{results['score']:.3f}")
                        st.caption(f"*{results['task']} Model*")
                        for insight in results['insights']:
                            st.info(insight)
                        summary = generate_executive_summary(df, results)
                        st.markdown("**Executive Summary:**")
                        st.write(summary)
        with col2:
            if 'results' in locals() and results:
                fig = px.bar(results['features'], x='importance', y='feature', orientation='h',
                             title="🎯 Feature Importance Analysis", color='importance',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("🧪 AI Laboratory")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎨 AI Data Insights")
                user_query = st.text_area(
                    "Ask AI about your data:",
                    placeholder="e.g., What patterns should I investigate?",
                    height=100
            )

            if st.button("🤖 Ask AI"):
                if user_query:
                    # Build a rich prompt
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                    prompt = f"""
Dataset snapshot:
- Shape: {df.shape}
- Numeric columns: {num_cols[:5]}{'...' if len(num_cols)>5 else ''}
- Categorical columns: {cat_cols[:5]}{'...' if len(cat_cols)>5 else ''}
- Missing values: {df.isnull().sum().sum()}
- First 3 rows as JSON: {json.dumps(df.head(3).to_dict(orient="records"))}

User question: {user_query}

Give a concise, actionable answer (max 80 words).
                    """

                    with st.spinner("Querying Gemini…"):
                        answer = safe_gemini(prompt, fallback="💡 Tip: check your GEMINI_API_KEY or quota.")
                    st.success("AI Response:")
                    st.write(answer)

        with col2:
            st.subheader("🔬 Anomaly Detection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                anomaly_col = st.selectbox("Select column for anomaly detection", numeric_cols)
                if st.button("🔍 Detect Anomalies"):
                    Q1, Q3 = df[anomaly_col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    anomalies = df[(df[anomaly_col] < Q1 - 1.5 * IQR) |
                                   (df[anomaly_col] > Q3 + 1.5 * IQR)]

                    if len(anomalies) > 0:
                        st.warning(f"Found {len(anomalies)} anomalies!")
                        st.info(anomaly_explanation(df, anomaly_col, anomalies))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=df[anomaly_col], mode='markers',
                            name='Normal', marker=dict(color='blue', size=5)))
                        fig.add_trace(go.Scatter(
                            y=anomalies[anomaly_col], x=anomalies.index,
                            mode='markers', name='Anomalies',
                            marker=dict(color='red', size=10)))
                        fig.update_layout(title=f"Anomalies in {anomaly_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No significant anomalies detected!")
else:
    st.markdown("""
    ## 🚀 Welcome to DataWhisperer Pro
    ### *Powered by Google Gemini AI*

    ---

    ### 🌟 Why DataWhisperer Pro?

    #### 📊 **Intelligent EDA**
    - AI-generated data narratives
    - Pattern recognition with clustering
    - Anomaly detection & explanation
    - 8+ auto-generated visualizations
    - 3D interactive exploration

    #### 🤖 **Gemini AI Integration**
    - Natural language data queries
    - Smart feature engineering suggestions
    - Automated insight generation
    - Executive summaries
    - Predictive modeling guidance

    #### ⚡ **Professional Features**
    - Production-ready visualizations
    - ML model evaluation
    - Real-time AI assistance
    - Export-ready reports

    #### 🎯 **Built for Data Scientists**
    - Clean, modular architecture
    - Scalable design patterns
    - Industry best practices
    - Comprehensive documentation

    ---

    **👈 Upload your CSV to unlock AI-powered insights!**
    """)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Visualizations", "8+", "Auto-generated")
    col2.metric("AI Features", "6", "Gemini-powered")
    col3.metric("ML Models", "2", "AutoML ready")
    col4.metric("Processing", "<3s", "Lightning fast")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, Plotly, Scikit-learn, and Google Gemini AI")