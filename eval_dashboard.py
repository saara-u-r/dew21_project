import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import json
import os
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from typing import Any

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DEW21 RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewBlockContainer"] {
    background-color: #0d0d0d !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #f57c00 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    color: #999999 !important;
    font-weight: 500 !important;
}

/* Card styling for plots */
.plot-container {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #333;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.hero-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 24px;
    color: #fff;
    text-align: center;
}
.hero-logo span { color: #f57c00; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    csv_path = "evaluation/eval_latest.csv"
    json_path = "evaluation/eval_latest.json"
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return None, None
        
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        summary = json.load(f)
        
    return df, summary

df, summary = load_data()

# --- HEADER ---
st.markdown('<div class="hero-logo">DEW<span>21</span> <i>"Pitch Perfect"</i> Eval Engine</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 40px;'>Totally real, definitely not cherry-picked AI Performance Metrics</p>", unsafe_allow_html=True)

if df is None:
    st.error("⚠️ No evaluation data found! Please run the evaluation suite first (`python -m src.evaluate_rag`).")
    st.stop()

assert df is not None  # type: ignore

st.sidebar.markdown("### 🎛️ Pitch Controls")
selected_domain = st.sidebar.radio("Knowledge Domain", ["All Domains 🌍", "⚡ Electricity", "🔥 Natural Gas"])
if "Domain" in df.columns:
    if selected_domain == "⚡ Electricity":
        df = df[df["Domain"] == "Electricity"]
    elif selected_domain == "🔥 Natural Gas":
        df = df[df["Domain"] == "Natural Gas"]
    
if df.empty:
    st.warning("No data for this domain combination!")
    st.stop()

# --- SCORECARD (Top Impact Numbers) ---
st.markdown("### 💸 The 'Please Fund Us' Executive Scorecard")

# Calculate metrics
avg_latency = df["Latency_s"].mean()
avg_correctness = df["correctness_score"].mean() * 100
avg_faithfulness = df["faithfulness_score"].mean() * 100
avg_hallucination = df["hallucination_score"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Average Correctness", value=f"{avg_correctness:.1f}%", help="How accurately the RAG matches the ground truth.")
with col2:
    st.metric(label="Average Faithfulness", value=f"{avg_faithfulness:.1f}%", help="How strictly the RAG relies only on retrieved context.")
with col3:
    # Inverse color for hallucination
    st.metric(label="Hallucination Rate", value=f"{avg_hallucination:.1f}%", help="Lower is better. Measures ungrounded claims.")
with col4:
    st.metric(label="Average Latency", value=f"{avg_latency:.1f}s", help="Time taken from query to final answer.")

st.markdown("<br>", unsafe_allow_html=True)

# --- CHARTS SECTION ---
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 🕸️ The 'Look How Smart Our AI Is' Spiderweb")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Prepare data for Radar Chart
    metrics = ['Faithfulness', 'Relevance', 'Context Precision', 'Context Recall', 'Correctness', 'Hallucination']
    # We invert hallucination for the radar chart so "further out" is always better.
    # Actually, standard radar charts map the raw value. Let's just plot the 0-1 values.
    # To make the shape look good if hallucination is low, we can plot (1 - hallucination) and label it "Factual Accuracy", 
    # but the prompt specifically asked for the hallucination metric. Let's just map it.
    
    values = [
        df["faithfulness_score"].mean(),
        df["relevance_score"].mean(),
        df["context_precision_score"].mean(),
        df["context_recall_score"].mean(),
        df["correctness_score"].mean(),
        df["hallucination_score"].mean()
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        fillcolor='rgba(156, 39, 176, 0.4)', # Vivid Purple for Pitch!
        line={'color': '#e1bee7', 'width': 3},
        name='Averages'
    ))
    
    fig_radar.update_layout(
        polar={
            'bgcolor': 'rgba(0,0,0,0)',
            'radialaxis': {
                'visible': True,
                'range': [0, 1],
                'gridcolor': '#333',
                'linecolor': '#333',
                'tickfont': {'color': '#aaa'}
            },
            'angularaxis': {
                'gridcolor': '#333',
                'linecolor': '#333',
                'tickfont': {'color': '#eee', 'size': 12}
            }
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown("### 📊 Where We Shine (And Where We Fake It)")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Group by category and calculate means
    cat_df = df.groupby("Category")[["correctness_score", "faithfulness_score", "relevance_score"]].mean().reset_index()
    cat_df = cat_df.rename(columns={"correctness_score": "Correctness", "faithfulness_score": "Faithfulness", "relevance_score": "Relevance"})
    
    # Melt for plotly express
    melted_df = pd.melt(cat_df, id_vars=['Category'], value_vars=['Correctness', 'Faithfulness', 'Relevance'], 
                        var_name='Metric', value_name='Score')
    
    fig_bar = px.bar(
        melted_df, x='Category', y='Score', color='Metric', barmode='group',
        color_discrete_sequence=['#f57c00', '#29b6f6', '#9c27b0'], title="Knowledge Domain Mastery"
    )
    
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#eee'},
        xaxis={'title': "Knowledge Domain", 'showgrid': False, 'linecolor': '#333'},
        yaxis={'title': "Score (0 to 1)", 'showgrid': True, 'gridcolor': '#333', 'range': [0, 1]},
        legend={'title': "", 'orientation': "h", 'yanchor': "bottom", 'y': 1.02, 'xanchor': "right", 'x': 1},
        margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- DRILL DOWN EXPANDER ---
st.markdown("### 🔍 Receipts: LLM Judge Interrogation Log")
with st.expander("Peek behind the curtain at exactly how strict our LLM Evaluator is"):
    
    # Category filter
    selected_cat_filter = st.selectbox("Show off our answers in:", ["All Topics"] + list(df["Category"].unique()))
    
    filtered_df = df if selected_cat_filter == "All Topics" else df[df["Category"] == selected_cat_filter]
    
    for idx, row in filtered_df.iterrows():
        domain_str = f"| {row['Domain']} " if "Domain" in row else ""
        st.markdown(f"**Q: {row['Question']}**  *(Category: {row['Category']} {domain_str})*")
        
        # Mini scorecard for this question
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Faithfulness", f"{row['faithfulness_score']:.1f}")
        sc2.metric("Relevance", f"{row['relevance_score']:.1f}")
        sc3.metric("Precision", f"{row['context_precision_score']:.1f}")
        sc4.metric("Recall", f"{row['context_recall_score']:.1f}")
        sc5.metric("Correctness", f"{row['correctness_score']:.1f}")
        sc6.metric("Hallucination", f"{row['hallucination_score']:.1f}")
        
        # If hallucination > 0.3, show a warning badge
        if row['hallucination_score'] > 0.3:
            st.warning(f"⚠️ **Judge noted hallucination:** {row.get('hallucination_reason', 'N/A')}")
            
        st.markdown("**Answer Preview:**")
        st.info(row['Answer_Preview'] + "...")
        st.divider()
