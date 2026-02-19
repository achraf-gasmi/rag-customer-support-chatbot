import streamlit as st
import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))
from src.backend import get_response, load_knowledge_base

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEase Support",
    page_icon="🛍️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #1a1a2e; }

    /* User bubble - RIGHT side */
    .chat-user {
        background-color: #0084ff;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 4px 0 4px auto;
        max-width: 60%;
        width: fit-content;
        display: block;
        text-align: right;
        margin-left: 40%;
    }

    /* Bot bubble - LEFT side */
    .chat-bot {
        background-color: #2a2a3e;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 4px 0;
        max-width: 60%;
        width: fit-content;
        display: block;
        border: 1px solid #3a3a5e;
    }

    /* Metric cards */
    .metric-card {
        background-color: #2a2a3e;
        border: 1px solid #3a3a5e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: #e0e0e0;
        font-size: 0.9em;
    }
    .metric-card b {
        display: block;
        font-size: 1.6em;
        color: #ffffff;
        margin: 4px 0;
    }

    /* Confidence colors */
    .conf-high { color: #4cff91 !important; }
    .conf-mid  { color: #ffc107 !important; }
    .conf-low  { color: #ff4c4c !important; }

    /* Source badges */
    .badge-kb {
        background-color: #1a4731;
        color: #4cff91;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.82em;
        font-weight: bold;
    }
    .badge-llm {
        background-color: #1a2f4e;
        color: #4ca3ff;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.82em;
        font-weight: bold;
    }

    /* Input box */
    .stTextInput input {
        background-color: #2a2a3e !important;
        color: #ffffff !important;
        border: 1px solid #3a3a5e !important;
        border-radius: 12px !important;
    }

    /* Send button */
    .stFormSubmitButton button {
        background-color: #0084ff !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 8px 24px !important;
        font-weight: bold !important;
    }

    /* Divider */
    hr { border-color: #3a3a5e; }

    /* Sidebar text */
    .stMarkdown p, .stMarkdown h3 { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "kb_loaded" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        load_knowledge_base()
    st.session_state.kb_loaded = True

# ── Header ────────────────────────────────────────────────────────────
st.markdown("## 🛍️ ShopEase Customer Support")
st.markdown("Ask me anything about your orders, refunds, shipping, and more.")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────
col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    # Chat history
    chat_container = st.container()
    with chat_container:
        for entry in st.session_state.chat_log:
            st.markdown(
                f'<div class="chat-user">🧑 {entry["query"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin: 2px 0'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-bot">🤖 {entry["response"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin: 10px 0'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message",
            placeholder="e.g. How do I cancel my order?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send 📨")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            result = get_response(user_input, st.session_state.conversation_history)

        st.session_state.chat_log.append({
            "query":            user_input,
            "response":         result["response"],
            "confidence_score": result["confidence_score"],
            "source":           result["source"],
            "category":         result["category"],
            "intent":           result["intent"],
            "timestamp":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        st.rerun()

with col_sidebar:
    st.markdown("### 📊 Session Analytics")

    if st.session_state.chat_log:
        total     = len(st.session_state.chat_log)
        kb_count  = sum(1 for e in st.session_state.chat_log if e["source"] == "knowledge_base")
        llm_count = total - kb_count
        avg_conf  = sum(e["confidence_score"] for e in st.session_state.chat_log) / total

        # Metric cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    💬<b>{total}</b>Queries
                </div>""", unsafe_allow_html=True)
        with c2:
            conf_class = "conf-high" if avg_conf >= 0.75 else "conf-mid" if avg_conf >= 0.5 else "conf-low"
            st.markdown(f"""
                <div class="metric-card">
                    🎯<b class="{conf_class}">{avg_conf:.2f}</b>Avg Confidence
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
                <div class="metric-card">
                    📚<b>{kb_count}</b>From KB
                </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="metric-card">
                    🤖<b>{llm_count}</b>From LLM
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🕓 Last Response Info")

        last       = st.session_state.chat_log[-1]
        conf       = last["confidence_score"]
        conf_class = "conf-high" if conf >= 0.75 else "conf-mid" if conf >= 0.5 else "conf-low"
        badge      = "badge-kb" if last["source"] == "knowledge_base" else "badge-llm"
        badge_text = "📚 Knowledge Base" if last["source"] == "knowledge_base" else "🤖 LLM Generated"

        st.markdown(f'**Source:** <span class="{badge}">{badge_text}</span>', unsafe_allow_html=True)
        st.markdown(f'**Confidence:** <span class="{conf_class}">{conf}</span>', unsafe_allow_html=True)
        st.markdown(f"**Category:** `{last['category']}`")
        st.markdown(f"**Intent:** `{last['intent']}`")
        st.markdown(f"**Time:** `{last['timestamp']}`")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Confidence Over Time")
        st.line_chart(
            {"Confidence": [e["confidence_score"] for e in st.session_state.chat_log]}
        )

    else:
        st.info("💡 Start chatting to see analytics here!")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.chat_log             = []
        st.rerun()