"""
Streamlit Dashboard for Hybrid ESN-Ridge Stock Predictor

Matching the Vue/React app design with dark theme and modern UI.
"""
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
USE_API = os.environ.get("USE_API", "true").lower() == "true"

# Page config
st.set_page_config(
    page_title="Hybrid ESN-Ridge Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching Vue app design
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(to bottom right, #020617, #0f172a, #000000);
        color: #f1f5f9;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom header */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Dark card styling */
    .dark-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.25rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    }
    
    .dark-card-large {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #1e293b;
        border-radius: 1.5rem;
        padding: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }
    
    /* Label styling */
    .label-text {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.3em;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    
    /* Value styling */
    .value-text {
        font-size: 2.25rem;
        font-weight: 600;
        color: #f1f5f9;
        letter-spacing: 0.05em;
    }
    
    /* Horizon card */
    .horizon-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.25rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        min-width: 240px;
    }
    
    .horizon-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        opacity: 0.6;
    }
    
    .horizon-h1::before { background: linear-gradient(90deg, transparent, #60a5fa, transparent); }
    .horizon-h5::before { background: linear-gradient(90deg, transparent, #f59e0b, transparent); }
    .horizon-h20::before { background: linear-gradient(90deg, transparent, #10b981, transparent); }
    
    /* News card */
    .news-card {
        background: rgba(2, 6, 23, 0.6);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1);
    }
    
    /* Chat panel - white background */
    .chat-panel {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #cbd5e1;
        border-radius: 1.5rem;
        padding: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        color: #000000;
    }
    
    .chat-message-user {
        background: #dbeafe;
        border: 1px solid #93c5fd;
        border-radius: 1rem;
        padding: 1rem;
        margin-left: auto;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message-bot {
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        border-radius: 1rem;
        padding: 1rem;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar */
    .progress-container {
        margin-top: 1rem;
    }
    
    .progress-label {
        font-size: 0.75rem;
        color: #64748b;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    
    .progress-bar {
        height: 8px;
        width: 100%;
        background: #1e293b;
        border-radius: 9999px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s ease;
    }
    
    /* Button styling */
    .stButton > button {
        background: #10b981;
        color: #000000;
        border: none;
        border-radius: 1rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #059669;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.25rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Trend badge */
    .trend-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-top: 0.5rem;
    }
    
    .trend-dot {
        height: 6px;
        width: 6px;
        border-radius: 50%;
        background: #6ee7b7;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(2, 6, 23, 0.6);
        border: 1px solid rgba(30, 41, 59, 0.6);
        border-radius: 1rem;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Text colors */
    .text-slate-400 { color: #94a3b8; }
    .text-slate-500 { color: #64748b; }
    .text-emerald-300 { color: #6ee7b7; }
    
    /* Override Streamlit text colors */
    .stMarkdown { color: #f1f5f9; }
    h1, h2, h3 { color: #f1f5f9; }
    
    /* Sidebar styling */
    .css-1d391kg { background: rgba(15, 23, 42, 0.95); }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #cbd5e1;
        border-radius: 1rem;
        color: #000000;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #cbd5e1;
        border-radius: 1rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme matching Vue app
HORIZON_COLORS = {
    'h1': '#60a5fa',
    'h5': '#f59e0b',
    'h20': '#10b981'
}

HORIZON_LABELS = {
    'h1': '1-Day',
    'h5': '5-Day',
    'h20': '20-Day'
}


@st.cache_data(ttl=300)
def fetch_predictions_from_api() -> Optional[Dict[str, Any]]:
    """Fetch predictions from FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/predict", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch predictions from API: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_news_from_api(days_back: int = 7) -> list:
    """Fetch news from FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/news", params={"days_back": days_back}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return []


@st.cache_data(ttl=600)
def fetch_model_info_from_api() -> Optional[Dict[str, Any]]:
    """Fetch model information from FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def chat_with_rag(question: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """Send question to RAG chat endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get chat response: {e}")
        return None


def create_prediction_chart(df: pd.DataFrame) -> go.Figure:
    """Create chart matching Vue app style"""
    fig = go.Figure()
    
    # Dark theme colors
    bg_color = 'rgba(2, 6, 23, 0.6)'
    grid_color = '#1e293b'
    text_color = '#94a3b8'
    
    # Add lines for each horizon
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['h1_prediction'] * 100,
        mode='lines+markers',
        name='h1',
        line=dict(color=HORIZON_COLORS['h1'], width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['h5_prediction'] * 100,
        mode='lines+markers',
        name='h5',
        line=dict(color=HORIZON_COLORS['h5'], width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['h20_prediction'] * 100,
        mode='lines+markers',
        name='h20',
        line=dict(color=HORIZON_COLORS['h20'], width=3),
        marker=dict(size=7)
    ))
    
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color),
        xaxis=dict(
            gridcolor=grid_color,
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=grid_color,
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickformat='.2f%'
        ),
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color=text_color)
        )
    )
    
    return fig


def render_horizon_card(horizon: str, label: str, signal: str, prediction: float, color: str):
    """Render a horizon summary card"""
    intensity = abs(prediction) * 100
    progress_width = min(intensity, 100)
    
    st.markdown(f"""
    <div class="horizon-card horizon-{horizon}">
        <div class="label-text">{label}</div>
        <div class="value-text">
            {signal}
            <span style="font-size: 1rem; color: #94a3b8; margin-left: 0.5rem;">
                {(prediction * 100):.2f}%
            </span>
        </div>
        <div class="progress-container">
            <div class="progress-label">
                <span>Momentum</span>
                <span>{intensity:.2f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_width}%; background: {color};"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    
    # Fetch data
    if USE_API:
        data = fetch_predictions_from_api()
        if not data:
            st.error("Unable to fetch predictions. Please ensure the FastAPI backend is running.")
            st.info("Start the backend with: `cd application/backend && python main.py`")
            return
        
        predictions = data.get('predictions', [])
        recent_news = data.get('recent_news', [])
        symbol = data.get('symbol', 'SPY')
        generated_at = data.get('generated_at', '')
    else:
        st.error("Standalone mode not yet implemented. Please use API mode.")
        return
    
    if not predictions:
        st.warning("No predictions available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Latest prediction
    latest = df.iloc[-1]
    
    # Header Summary Cards (matching Vue app layout)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="dark-card">
            <div class="label-text">Symbol</div>
            <div class="value-text" style="letter-spacing: 0.1em;">{symbol}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        h20_signal = latest['h20_signal']
        h20_pred = latest['h20_prediction'] * 100
        actual_close = latest['actual_close']
        
        st.markdown(f"""
        <div class="dark-card">
            <div class="label-text">Latest h20 Signal</div>
            <div class="value-text" style="font-size: 1.875rem;">{h20_signal}</div>
            <div style="font-size: 0.875rem; color: #94a3b8; margin-top: 0.25rem;">
                {h20_pred:.2f}% · Close ${actual_close:.2f}
            </div>
            <div class="trend-badge">
                Trend
                <span class="trend-dot"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        last_updated = generated_at[:19] if generated_at else "Awaiting data"
        try:
            dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            last_updated = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        st.markdown(f"""
        <div class="dark-card">
            <div class="label-text">Last Updated</div>
            <div class="value-text" style="font-size: 1.25rem;">{last_updated}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Horizon Summary Cards
    st.markdown("### Horizon Summary")
    h1_col, h5_col, h20_col = st.columns(3)
    
    with h1_col:
        render_horizon_card('h1', HORIZON_LABELS['h1'], latest['h1_signal'], 
                           latest['h1_prediction'], HORIZON_COLORS['h1'])
    
    with h5_col:
        render_horizon_card('h5', HORIZON_LABELS['h5'], latest['h5_signal'], 
                           latest['h5_prediction'], HORIZON_COLORS['h5'])
    
    with h20_col:
        render_horizon_card('h20', HORIZON_LABELS['h20'], latest['h20_signal'], 
                           latest['h20_prediction'], HORIZON_COLORS['h20'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content Area - matching Vue app layout
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        # Chart Section
        st.markdown("""
        <div class="dark-card-large">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem;">Return Forecasts</h2>
                    <p style="font-size: 0.875rem; color: #94a3b8;">
                        Predicted % change · positive → BUY bias
                    </p>
                </div>
                <div style="display: flex; gap: 0.75rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.3em; color: #64748b;">
                    <span style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="height: 8px; width: 16px; border-radius: 2px; background: #60a5fa;"></span>
                        h1
                    </span>
                    <span style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="height: 8px; width: 16px; border-radius: 2px; background: #f59e0b;"></span>
                        h5
                    </span>
                    <span style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="height: 8px; width: 16px; border-radius: 2px; background: #10b981;"></span>
                        h20
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart container
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_prediction_chart(df)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent Headlines Section
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h2 style="font-size: 1.25rem; font-weight: 600;">Recent Headlines</h2>
                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.3em; color: #64748b;">
                    sentiment cues
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if recent_news:
            news_col1, news_col2 = st.columns(2)
            for idx, news in enumerate(recent_news):
                col = news_col1 if idx % 2 == 0 else news_col2
                with col:
                    date_str = news.get('date', '')[:16] if news.get('date') else ''
                    publisher = news.get('publisher', 'Source')
                    title = news.get('title', 'No title')
                    link = news.get('link', '#')
                    
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="display: flex; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.5rem;">
                            <h3 style="font-size: 1rem; font-weight: 600; line-height: 1.4; color: #f1f5f9;">{title}</h3>
                            <span style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.3em; color: #6ee7b7;">
                                {publisher}
                            </span>
                        </div>
                        <p style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem;">
                            {date_str}
                        </p>
                        {f'<a href="{link}" target="_blank" style="font-size: 0.75rem; color: #6ee7b7; text-decoration: none;">View source →</a>' if link != '#' else ''}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #94a3b8;">No recent headlines available.</p>', unsafe_allow_html=True)
    
    with main_col2:
        # Chat Panel - white background matching Vue app
        st.markdown("""
        <div class="chat-panel" style="height: 100%; display: flex; flex-direction: column;">
            <div style="margin-bottom: 1rem;">
                <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; color: #000000; letter-spacing: -0.02em;">
                    Post your questions here!
                </h2>
                <p style="font-size: 0.875rem; color: rgba(0,0,0,0.7);">
                    Ask about signals, risk posture, or key headlines. Responses come from the RAG layer.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat messages container
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.6); border: 1px solid rgba(203, 213, 225, 0.7); 
                    border-radius: 1rem; padding: 1rem; max-height: 400px; overflow-y: auto; margin-bottom: 1rem;">
        """, unsafe_allow_html=True)
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message-user" style="margin-bottom: 0.75rem;">
                    <div style="color: #000000; line-height: 1.6;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message-bot" style="margin-bottom: 0.75rem;">
                    <div style="color: #000000; line-height: 1.6;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if msg.get('context'):
                    with st.expander(f"View context sources ({len(msg['context'])})"):
                        for ctx in msg['context']:
                            st.markdown(f"**{ctx.get('title', 'Unknown')}**")
                            if ctx.get('score') is not None:
                                st.caption(f"Relevance: {ctx.get('score', 0)*100:.0f}%")
        
        # Quick prompts (only show if no messages)
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="color: rgba(0,0,0,0.6); font-size: 0.875rem;">
                <p style="margin-bottom: 0.75rem;">Ask the assistant for tailored insight—try one of these quick prompts to get started:</p>
            </div>
            """, unsafe_allow_html=True)
            
            quick_prompts = [
                "Summarize today's cross-horizon signals with supporting news.",
                "What is driving the latest SELL signal on h5?",
                "List the strongest bullish headlines in the last 48 hours."
            ]
            
            # Store selected prompt in session state
            if 'selected_prompt' not in st.session_state:
                st.session_state.selected_prompt = None
            
            prompt_cols = st.columns(len(quick_prompts))
            for idx, prompt in enumerate(quick_prompts):
                with prompt_cols[idx]:
                    if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input area
        user_input = st.text_area(
            "",
            key="chat_input",
            height=60,
            label_visibility="collapsed",
            placeholder="Type your query... Press Enter to send"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            send_clicked = st.button("Send", use_container_width=True, type="primary")
        with col_btn2:
            clear_clicked = st.button("Clear", use_container_width=True)
        
        # Handle quick prompt or user input
        query = None
        if st.session_state.get('selected_prompt'):
            query = st.session_state.selected_prompt
            st.session_state.selected_prompt = None  # Clear after use
        elif send_clicked and user_input:
            query = user_input
        
        if query:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query
            })
            
            # Get response
            with st.spinner("Generating insight..."):
                response = chat_with_rag(query, top_k=3)
            
            if response:
                answer = response.get('answer', 'No answer available')
                context = response.get('context', [])
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'context': context
                })
            
            st.rerun()
        
        if clear_clicked:
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
