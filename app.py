"""
app.py — Streamlit UI for Hinglish Sentiment Analyzer
Run: streamlit run app.py
"""

import os
import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_utils import predict, get_token_importance, LABELS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hinglish Sentiment Analyzer",
    page_icon="🌐",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .title { font-size: 2.2rem; font-weight: 800; color: #ffffff; text-align: center; }
    .subtitle { font-size: 1rem; color: #9ca3af; text-align: center; margin-bottom: 2rem; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .positive  { background-color: #052e16; border-left: 5px solid #22c55e; color: #86efac; }
    .negative  { background-color: #450a0a; border-left: 5px solid #ef4444; color: #fca5a5; }
    .neutral   { background-color: #1c1917; border-left: 5px solid #f59e0b; color: #fcd34d; }
    .token-box {
        display: inline-block;
        margin: 3px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .example-btn { cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🌐 Hinglish Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time sentiment analysis for code-mixed Hindi-English text · Powered by multilingual BERT</div>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
SAVE_DIR = "saved_model"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    if not os.path.exists(SAVE_DIR):
        st.error("⚠️ No trained model found. Please run `python train.py` first.")
        st.stop()
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

# ── Example inputs ────────────────────────────────────────────────────────────
EXAMPLES = [
    "yaar ye movie ekdum bakwaas thi, waste of time",
    "bhai kal ka concert toh mast tha, life mein best experience!",
    "thoda theek tha, kuch khaas nahi tha actually",
    "itna bura experience nahi socha tha, bohot disappoint hua",
    "aaj ka din bahut achha gaya, sab kuch perfect raha!",
]

st.markdown("#### 💬 Try an example")
cols = st.columns(len(EXAMPLES))
selected_example = None
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    with col:
        if st.button(f"#{i+1}", use_container_width=True, help=ex):
            selected_example = ex

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("#### ✍️ Or enter your own Hinglish text")
default_text = selected_example if selected_example else ""
user_input = st.text_area(
    label="Input",
    value=default_text,
    placeholder="e.g. yaar ye movie ekdum bakwaas thi lol",
    height=100,
    label_visibility="collapsed"
)

analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("Analyzing..."):
        label, confidence, all_scores = predict(user_input, model, tokenizer, DEVICE)
        token_importance               = get_token_importance(user_input, model, tokenizer, DEVICE)

    # Result box
    css_class = label.lower()
    emoji     = {"Positive": "😄", "Negative": "😠", "Neutral": "😐"}[label]
    st.markdown(
        f'<div class="result-box {css_class}">{emoji} {label} &nbsp;·&nbsp; {confidence*100:.1f}% confidence</div>',
        unsafe_allow_html=True
    )

    # Confidence bar chart
    st.markdown("#### 📊 Confidence Scores")
    colors = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"}
    fig = go.Figure(go.Bar(
        x=list(all_scores.values()),
        y=list(all_scores.keys()),
        orientation="h",
        marker_color=[colors[l] for l in all_scores.keys()],
        text=[f"{v*100:.1f}%" for v in all_scores.values()],
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=180
    )
    st.plotly_chart(fig, use_container_width=True)

    # Token importance
    if token_importance:
        st.markdown("#### 🔍 Word Influence (what drove the prediction)")
        html_tokens = ""
        for token, score in token_importance:
            # Green tint for high importance, grey for low
            intensity = int(score * 200)
            bg    = f"rgba(34, 197, 94, {score:.2f})" if label == "Positive" else \
                    f"rgba(239, 68, 68, {score:.2f})"  if label == "Negative" else \
                    f"rgba(245, 158, 11, {score:.2f})"
            color = "#ffffff" if score > 0.4 else "#9ca3af"
            html_tokens += f'<span class="token-box" style="background:{bg}; color:{color};">{token}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        st.caption("Darker = stronger influence on prediction")

elif analyze_btn and not user_input.strip():
    st.warning("Please enter some text first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "Built by <a href='https://github.com/SinchanaDevadi' style='color:#6b7280;'>Sinchana H Devadiga</a> · "
    "Model: bert-base-multilingual-cased · "
    "<a href='https://github.com/SinchanaDevadi/hinglish-sentiment-analyzer' style='color:#6b7280;'>View on GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
