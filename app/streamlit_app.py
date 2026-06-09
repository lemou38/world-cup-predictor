"""
FIFA World Cup 2026 Predictor - Premium Streamlit Application
Dark gold theme with animated visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import time
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Page config MUST be first
st.set_page_config(
    page_title="⚽ FIFA World Cup 2026 AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PREMIUM CSS THEME ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --gold: #C9A84C;
    --gold-light: #E8C96B;
    --gold-dark: #8B6914;
    --bg-deep: #050810;
    --bg-panel: #0D1117;
    --bg-card: #131920;
    --bg-elevated: #1a2332;
    --text-primary: #F0EAD6;
    --text-muted: #8B9BB4;
    --green: #2ECC71;
    --red: #E74C3C;
    --blue: #3498DB;
    --border: rgba(201,168,76,0.2);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

/* Main background */
.stApp { background-color: var(--bg-deep) !important; }
.main .block-container { 
    padding: 1.5rem 2rem 3rem 2rem !important; 
    max-width: 1400px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1a 0%, #050810 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #050810 0%, #0d1a2e 40%, #1a2d00 100%);
    border: 1px solid var(--gold-dark);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(201,168,76,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    color: var(--gold-light);
    letter-spacing: 4px;
    margin: 0;
    line-height: 1;
    text-shadow: 0 0 40px rgba(201,168,76,0.4);
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Countdown */
.countdown-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1rem 0;
}
.countdown-unit {
    background: rgba(201,168,76,0.08);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.countdown-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: var(--gold-light);
    line-height: 1;
    display: block;
}
.countdown-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--gold-light);
}

/* Match prediction card */
.match-card {
    background: var(--bg-card);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 2px;
    color: #fff;
}
.vs-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: var(--gold);
    text-align: center;
    padding: 0.5rem 0;
}
.score-display {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: var(--gold-light);
    text-align: center;
    letter-spacing: 4px;
}

/* Probability bars */
.prob-container { margin: 1.2rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
}
.prob-bar-fill-home {
    background: linear-gradient(90deg, #2ECC71, #27AE60);
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}
.prob-bar-fill-draw {
    background: linear-gradient(90deg, #F39C12, #E67E22);
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}
.prob-bar-fill-away {
    background: linear-gradient(90deg, #E74C3C, #C0392B);
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* Section headers */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: var(--gold-light);
    letter-spacing: 3px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1.2rem 0;
}

/* Tournament bracket */
.bracket-match {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 4px 0;
    font-size: 0.85rem;
}
.bracket-winner {
    color: var(--gold-light);
    font-weight: 600;
}
.bracket-loser { color: var(--text-muted); }

/* Selectboxes */
div[data-testid="stSelectbox"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold-dark), var(--gold)) !important;
    color: #050810 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(201,168,76,0.5) !important;
}

/* Tab styling */
div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 1px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold-light) !important;
    border-bottom-color: var(--gold) !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, var(--gold-dark), var(--gold-light)) !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
div[data-testid="stMetricValue"] {
    color: var(--gold-light) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* Flag emoji sizing */
.flag { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load or train the predictor."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
    
    if not os.path.exists(model_path):
        st.info("🔧 First run: training AI model... this takes ~30 seconds")
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "src", "train.py")],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Try inline training
            os.chdir(os.path.join(os.path.dirname(__file__), ".."))
            from train import train
            train()
    
    from predict import MatchPredictor
    p = MatchPredictor()
    p.load()
    return p


@st.cache_data
def load_match_data():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    df_path = os.path.join(model_dir, "matches_df.pkl")
    if os.path.exists(df_path):
        return joblib.load(df_path)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "model_comparison.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


@st.cache_data
def load_elo_history():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "elo_history.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


# Country to flag emoji mapping
FLAG_MAP = {
    "Brazil": "🇧🇷", "Germany": "🇩🇪", "Italy": "🇮🇹", "Argentina": "🇦🇷",
    "France": "🇫🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Uruguay": "🇺🇾",
    "Netherlands": "🇳🇱", "Portugal": "🇵🇹", "Belgium": "🇧🇪", "Croatia": "🇭🇷",
    "Denmark": "🇩🇰", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Mexico": "🇲🇽",
    "Colombia": "🇨🇴", "Chile": "🇨🇱", "Poland": "🇵🇱", "USA": "🇺🇸",
    "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Nigeria": "🇳🇬", "Cameroon": "🇨🇲",
    "Ghana": "🇬🇭", "Ivory Coast": "🇨🇮", "South Korea": "🇰🇷", "Japan": "🇯🇵",
    "Iran": "🇮🇷", "Australia": "🇦🇺", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦",
    "Algeria": "🇩🇿", "Tunisia": "🇹🇳", "Egypt": "🇪🇬", "Serbia": "🇷🇸",
    "Czech Republic": "🇨🇿", "Austria": "🇦🇹", "Turkey": "🇹🇷", "Ukraine": "🇺🇦",
    "Ecuador": "🇪🇨", "Peru": "🇵🇪", "Paraguay": "🇵🇾", "Costa Rica": "🇨🇷",
    "Canada": "🇨🇦", "Norway": "🇳🇴", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Russia": "🇷🇺",
    "China": "🇨🇳", "New Zealand": "🇳🇿", "Indonesia": "🇮🇩", "Panama": "🇵🇦",
}


def get_flag(team: str) -> str:
    return FLAG_MAP.get(team, "🏳️")


# ─── COUNTDOWN TIMER ──────────────────────────────────────────────────────────
def render_countdown():
    wc_start = datetime(2026, 6, 11, 18, 0, 0, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = wc_start - now
    
    if delta.total_seconds() <= 0:
        st.markdown('<p style="color:#C9A84C;font-size:1.2rem;">🏆 The World Cup is LIVE!</p>', unsafe_allow_html=True)
        return
    
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    st.markdown(f"""
    <div class="countdown-grid">
        <div class="countdown-unit">
            <span class="countdown-number">{days:03d}</span>
            <div class="countdown-label">Days</div>
        </div>
        <div class="countdown-unit">
            <span class="countdown-number">{hours:02d}</span>
            <div class="countdown-label">Hours</div>
        </div>
        <div class="countdown-unit">
            <span class="countdown-number">{minutes:02d}</span>
            <div class="countdown-label">Minutes</div>
        </div>
        <div class="countdown-unit">
            <span class="countdown-number">{seconds:02d}</span>
            <div class="countdown-label">Seconds</div>
        </div>
    </div>
    <p style="color:var(--text-muted);font-size:0.8rem;text-align:center;letter-spacing:1px;">
        UNTIL FIFA WORLD CUP 2026 KICK-OFF · USA / CANADA / MEXICO
    </p>
    """, unsafe_allow_html=True)


# ─── GAUGE CHART ──────────────────────────────────────────────────────────────
def make_confidence_gauge(confidence: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "AI Confidence", "font": {"color": "#8B9BB4", "size": 13}},
        number={"suffix": "%", "font": {"color": "#C9A84C", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B9BB4", "tickfont": {"color": "#8B9BB4"}},
            "bar": {"color": "#C9A84C"},
            "bgcolor": "#131920",
            "bordercolor": "rgba(201,168,76,0.3)",
            "steps": [
                {"range": [0, 40], "color": "#1a2332"},
                {"range": [40, 70], "color": "#1e2d3d"},
                {"range": [70, 100], "color": "#1f3420"},
            ],
            "threshold": {
                "line": {"color": "#E8C96B", "width": 2},
                "thickness": 0.75,
                "value": confidence,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin={"t": 30, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F0EAD6"},
    )
    return fig


# ─── PROBABILITY BARS ─────────────────────────────────────────────────────────
def render_probability_bars(result: dict):
    ph = result["prob_home_win"]
    pd_ = result["prob_draw"]
    pa = result["prob_away_win"]
    ta = result["team_a"]
    tb = result["team_b"]
    fa = get_flag(ta)
    fb = get_flag(tb)

    # ── Header: Teams + Score ──
    col_a, col_mid, col_b = st.columns([2, 1, 2])
    with col_a:
        st.markdown(f"<div style='text-align:center;font-size:2.5rem;'>{fa}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:1.6rem;font-weight:800;letter-spacing:2px;color:#fff;'>{ta.upper()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:#8B9BB4;'>ELO: {result['elo_home']}</div>", unsafe_allow_html=True)
    with col_mid:
        st.markdown(f"<div style='text-align:center;font-size:2.2rem;font-weight:900;color:#C9A84C;padding-top:0.3rem;'>{result['predicted_score']}</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:0.75rem;color:#C9A84C;letter-spacing:2px;'>PREDICTED</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<div style='text-align:center;font-size:2.5rem;'>{fb}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:1.6rem;font-weight:800;letter-spacing:2px;color:#fff;'>{tb.upper()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:#8B9BB4;'>ELO: {result['elo_away']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Probability bars using native Streamlit ──
    st.markdown(f"**{fa} {ta} Win** — 🟢 {ph}%")
    st.progress(int(ph))

    st.markdown(f"**⚖️ Draw** — 🟡 {pd_}%")
    st.progress(int(pd_))

    st.markdown(f"**{fb} {tb} Win** — 🔴 {pa}%")
    st.progress(int(pa))


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # Load predictor
    with st.spinner("Loading AI model..."):
        try:
            predictor = load_predictor()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Run `python src/train.py` first, then refresh this page.")
            st.stop()
    
    teams = predictor.all_teams
    
    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:3rem;">⚽</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#C9A84C;letter-spacing:3px;">WC2026</div>
            <div style="font-size:0.7rem;color:#8B9BB4;letter-spacing:2px;text-transform:uppercase;">AI PREDICTOR</div>
        </div>
        <hr style="border-color:rgba(201,168,76,0.2);margin:0.5rem 0 1rem;">
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏟️ Match Prediction", "🏆 Tournament Simulator", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr style='border-color:rgba(201,168,76,0.2);'>", unsafe_allow_html=True)
        
        # Elo top 5 in sidebar
        st.markdown('<div style="font-size:0.75rem;color:#8B9BB4;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;">Top 5 by Elo</div>', unsafe_allow_html=True)
        top5 = sorted(predictor.elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (team, elo) in enumerate(top5):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.85rem;"><span>{medal} {get_flag(team)} {team}</span><span style="color:#C9A84C;">{elo:.0f}</span></div>', unsafe_allow_html=True)
    
    # ── HERO BANNER ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:3rem;">⚽</span>
            <div>
                <h1 class="hero-title">FIFA WORLD CUP 2026</h1>
                <p class="hero-subtitle">AI-Powered Match Prediction & Tournament Simulator</p>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <div style="font-size:0.7rem;color:#8B9BB4;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.3rem;">Powered by</div>
                <div style="font-size:0.85rem;color:#C9A84C;">XGBoost · ELO · Monte Carlo</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Countdown (always visible at top)
    with st.container():
        countdown_placeholder = st.empty()
        with countdown_placeholder.container():
            render_countdown()
    
    # ── PAGE ROUTING ─────────────────────────────────────────────────────────
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if "Match Prediction" in page:
        st.markdown('<div class="section-header">⚔️ Match Prediction</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            team_a = st.selectbox(
                "Team A (Home / Left)",
                teams,
                index=teams.index("Algeria") if "Algeria" in teams else 0,
                key="team_a"
            )
        
        with col2:
            st.markdown('<div style="text-align:center;padding-top:1.8rem;font-family:Bebas Neue,sans-serif;font-size:1.5rem;color:#C9A84C;">VS</div>', unsafe_allow_html=True)
            neutral = st.checkbox("Neutral venue", value=True)
        
        with col3:
            team_b = st.selectbox(
                "Team B (Away / Right)",
                [t for t in teams if t != team_a],
                index=[t for t in teams if t != team_a].index("Spain") if "Spain" in [t for t in teams if t != team_a] else 0,
                key="team_b"
            )
        
        predict_btn = st.button("⚽ PREDICT MATCH", use_container_width=True)
        
        if predict_btn or "last_prediction" in st.session_state:
            if predict_btn:
                with st.spinner("Running AI analysis..."):
                    result = predictor.predict(team_a, team_b, neutral=neutral)
                    st.session_state["last_prediction"] = result
            
            result = st.session_state.get("last_prediction")
            if result and result["team_a"] == team_a and result["team_b"] == team_b:
                render_probability_bars(result)
                
                col_g, col_stats = st.columns([1, 2])
                
                with col_g:
                    st.plotly_chart(
                        make_confidence_gauge(result["confidence"]),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                
                with col_stats:
                    st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
                    s1, s2 = st.columns(2)
                    with s1:
                        st.metric(f"{get_flag(team_a)} ELO", result["elo_home"])
                        st.metric("Expected Goals A", result["expected_home_goals"])
                    with s2:
                        st.metric(f"{get_flag(team_b)} ELO", result["elo_away"])
                        st.metric("Expected Goals B", result["expected_away_goals"])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Pie chart
                fig_pie = go.Figure(go.Pie(
                    labels=[f"{get_flag(team_a)} {team_a} Win", "Draw", f"{get_flag(team_b)} {team_b} Win"],
                    values=[result["prob_home_win"], result["prob_draw"], result["prob_away_win"]],
                    hole=0.5,
                    marker_colors=["#2ECC71", "#F39C12", "#E74C3C"],
                    textinfo="label+percent",
                    textfont_color="#F0EAD6",
                ))
                fig_pie.update_layout(
                    title={"text": "Outcome Probabilities", "font": {"color": "#C9A84C", "size": 14}},
                    height=280,
                    margin={"t": 40, "b": 0, "l": 0, "r": 0},
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend={"font": {"color": "#F0EAD6"}},
                    showlegend=True,
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif "Tournament Simulator" in page:
        st.markdown('<div class="section-header">🏆 World Cup 2026 Simulator</div>', unsafe_allow_html=True)
        
        from simulator import TournamentSimulator, WC2026_GROUPS
        
        col_cfg, col_run = st.columns([3, 1])
        
        with col_cfg:
            n_sims = st.slider("Monte Carlo simulations", 100, 2000, 500, 100)
            st.markdown(f'<div style="color:var(--text-muted);font-size:0.8rem;">Higher = more accurate, slower. ~{n_sims//100}s estimated</div>', unsafe_allow_html=True)
        
        with col_run:
            st.markdown("<div style='padding-top:1.5rem;'></div>", unsafe_allow_html=True)
            run_sim = st.button("🚀 RUN SIMULATION", use_container_width=True)
        
        # Display groups
        st.markdown('<div style="margin-top:1.5rem;margin-bottom:0.5rem;font-size:0.8rem;color:#8B9BB4;letter-spacing:2px;text-transform:uppercase;">WC2026 Groups (48 Teams · 12 Groups)</div>', unsafe_allow_html=True)
        
        group_cols = st.columns(6)
        for i, (g, teams_g) in enumerate(WC2026_GROUPS.items()):
            with group_cols[i % 6]:
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:0.6rem;margin-bottom:0.5rem;">
                    <div style="font-family:'Bebas Neue',sans-serif;color:#C9A84C;font-size:1rem;letter-spacing:2px;margin-bottom:0.4rem;">GROUP {g}</div>
                    {"".join(f'<div style="font-size:0.78rem;padding:2px 0;">{get_flag(t)} {t}</div>' for t in teams_g)}
                </div>
                """, unsafe_allow_html=True)
        
        if run_sim:
            progress_bar = st.progress(0)
            status = st.empty()
            status.markdown('<div style="color:#8B9BB4;font-size:0.85rem;">🔄 Running Monte Carlo simulations...</div>', unsafe_allow_html=True)
            
            sim = TournamentSimulator(predictor)
            
            results_container = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
            
            with st.spinner(f"Simulating {n_sims} tournaments..."):
                mc_results = sim.monte_carlo(n_sims, progress_callback=update_progress)
                single_run = sim.run_full_simulation()
            
            status.empty()
            progress_bar.empty()
            
            st.session_state["mc_results"] = mc_results
            st.session_state["single_run"] = single_run
        
        if "mc_results" in st.session_state:
            mc = st.session_state["mc_results"]
            run = st.session_state["single_run"]
            
            # Champion highlight
            champ_probs = mc["championship_probs"]
            top_team = max(champ_probs, key=champ_probs.get)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2d00,#0d1a00);border:2px solid #C9A84C;
                        border-radius:16px;padding:1.5rem;text-align:center;margin:1rem 0;">
                <div style="font-size:3rem;">{get_flag(top_team)}</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#E8C96B;letter-spacing:3px;">{top_team}</div>
                <div style="color:#8B9BB4;font-size:0.85rem;margin-top:0.3rem;">Most likely champion</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#C9A84C;">{champ_probs[top_team]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Championship probability chart
            top20 = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:20]
            labels = [f"{get_flag(t)} {t}" for t, _ in top20]
            vals = [v for _, v in top20]
            colors = [f"rgba(201,168,76,{0.4 + v/max(vals)*0.6})" for v in vals]
            
            fig_champ = go.Figure(go.Bar(
                x=vals,
                y=labels,
                orientation="h",
                marker_color=colors,
                marker_line_color="rgba(201,168,76,0.5)",
                marker_line_width=1,
                text=[f"{v:.1f}%" for v in vals],
                textposition="inside",
                textfont_color="#050810",
                textfont_size=11,
            ))
            fig_champ.update_layout(
                title={"text": f"🏆 Championship Probability ({mc['n_simulations']} simulations)", 
                       "font": {"color": "#C9A84C", "size": 14}},
                height=max(400, len(top20) * 28),
                margin={"t": 40, "b": 20, "l": 160, "r": 80},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis={"gridcolor": "rgba(255,255,255,0.05)", "tickcolor": "#8B9BB4", "tickfont": {"color": "#8B9BB4"}, "title": "Probability (%)"},
                yaxis={"tickcolor": "#8B9BB4", "tickfont": {"color": "#F0EAD6", "size": 12}},
            )
            st.plotly_chart(fig_champ, use_container_width=True, config={"displayModeBar": False})
            
            # Knockout bracket from single run
            st.markdown('<div class="section-header" style="font-size:1.3rem;">⚡ Sample Knockout Bracket</div>', unsafe_allow_html=True)
            
            rounds = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
            tabs = st.tabs(rounds)
            
            for i, rnd in enumerate(rounds):
                with tabs[i]:
                    matches_in_round = [m for m in run["knockout_bracket"] if m["round"] == rnd]
                    if not matches_in_round:
                        st.markdown('<div style="color:#8B9BB4;font-size:0.9rem;">No matches in this round</div>', unsafe_allow_html=True)
                        continue
                    
                    cols = st.columns(min(4, len(matches_in_round)))
                    for j, m in enumerate(matches_in_round):
                        with cols[j % len(cols)]:
                            t1_style = "color:#C9A84C;font-weight:700;" if m["winner"] == m["team1"] else "color:#8B9BB4;"
                            t2_style = "color:#C9A84C;font-weight:700;" if m["winner"] == m["team2"] else "color:#8B9BB4;"
                            st.markdown(f"""
                            <div class="bracket-match">
                                <div style="{t1_style}font-size:0.9rem;">{get_flag(m["team1"])} {m["team1"]}</div>
                                <div style="color:rgba(201,168,76,0.4);font-size:0.7rem;text-align:center;padding:2px 0;">vs</div>
                                <div style="{t2_style}font-size:0.9rem;">{get_flag(m["team2"])} {m["team2"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif "Analytics" in page:
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        df = load_match_data()
        elo_history = load_elo_history()
        model_info = load_model_comparison()
        
        if df.empty:
            st.warning("No match data found. Run training first.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Goals Distribution", "🏅 Top Nations", "🤖 Model Performance", "📉 Elo Evolution"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                total_goals = df["home_goals"] + df["away_goals"]
                fig_goals = px.histogram(
                    total_goals, nbins=15,
                    title="Total Goals per Match Distribution",
                    labels={"value": "Goals", "count": "Matches"},
                    color_discrete_sequence=["#C9A84C"],
                )
                fig_goals.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F0EAD6", title_font_color="#C9A84C",
                    xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                )
                st.plotly_chart(fig_goals, use_container_width=True, config={"displayModeBar": False})
            
            with c2:
                stage_goals = df.groupby("stage")[["home_goals", "away_goals"]].mean().round(2)
                fig_stage = go.Figure()
                fig_stage.add_trace(go.Bar(name="Home Goals", x=stage_goals.index, y=stage_goals["home_goals"], marker_color="#2ECC71"))
                fig_stage.add_trace(go.Bar(name="Away Goals", x=stage_goals.index, y=stage_goals["away_goals"], marker_color="#E74C3C"))
                fig_stage.update_layout(
                    title={"text": "Avg Goals by Stage", "font": {"color": "#C9A84C"}},
                    barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F0EAD6", xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                    legend={"font": {"color": "#F0EAD6"}},
                )
                st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})
        
        with tab2:
            # Top winning teams
            wins = {}
            for _, row in df.iterrows():
                if row["home_goals"] > row["away_goals"]:
                    wins[row["home_team"]] = wins.get(row["home_team"], 0) + 1
                elif row["away_goals"] > row["home_goals"]:
                    wins[row["away_team"]] = wins.get(row["away_team"], 0) + 1
            
            top15 = sorted(wins.items(), key=lambda x: x[1], reverse=True)[:15]
            labels = [f"{get_flag(t)} {t}" for t, _ in top15]
            vals = [v for _, v in top15]
            
            fig_top = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=[f"rgba(201,168,76,{0.4 + v/max(vals)*0.6})" for v in vals],
                text=vals, textposition="outside", textfont_color="#C9A84C",
            ))
            fig_top.update_layout(
                title={"text": "🏅 Most Wins (All Competitions)", "font": {"color": "#C9A84C", "size": 14}},
                height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F0EAD6",
                xaxis={"tickangle": -35, "tickfont": {"size": 11}, "gridcolor": "rgba(255,255,255,0.05)"},
                yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            )
            st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})
        
        with tab3:
            if model_info and "results" in model_info:
                model_results = model_info["results"]
                models_list = list(model_results.keys())
                accs = [model_results[m]["accuracy"] for m in models_list]
                f1s = [model_results[m]["f1"] for m in models_list]
                lls = [model_results[m]["log_loss"] for m in models_list]
                
                fig_models = go.Figure()
                fig_models.add_trace(go.Bar(name="Accuracy", x=models_list, y=accs, marker_color="#2ECC71"))
                fig_models.add_trace(go.Bar(name="F1 Score", x=models_list, y=f1s, marker_color="#3498DB"))
                fig_models.update_layout(
                    title={"text": "🤖 Model Comparison", "font": {"color": "#C9A84C", "size": 14}},
                    barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F0EAD6", legend={"font": {"color": "#F0EAD6"}},
                    xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.05)", "range": [0, 1]},
                )
                st.plotly_chart(fig_models, use_container_width=True, config={"displayModeBar": False})
                
                st.markdown(f'<div style="color:#C9A84C;font-size:0.9rem;margin-top:0.5rem;">✓ Best model selected: <strong>{model_info.get("name","N/A")}</strong></div>', unsafe_allow_html=True)
            else:
                st.info("Train the model to see comparison results.")
        
        with tab4:
            featured = ["Brazil", "Germany", "France", "Spain", "Argentina", "Algeria", "Morocco"]
            featured = [t for t in featured if t in elo_history]
            
            if elo_history and featured:
                fig_elo = go.Figure()
                colors = ["#C9A84C", "#2ECC71", "#3498DB", "#E74C3C", "#9B59B6", "#1ABC9C", "#E67E22"]
                
                for i, team in enumerate(featured):
                    records = elo_history[team]
                    years = [r["year"] for r in records[::5]]  # Subsample
                    elos = [r["elo"] for r in records[::5]]
                    
                    fig_elo.add_trace(go.Scatter(
                        x=years, y=elos, mode="lines",
                        name=f"{get_flag(team)} {team}",
                        line={"color": colors[i % len(colors)], "width": 2},
                    ))
                
                fig_elo.update_layout(
                    title={"text": "📉 Elo Rating Evolution", "font": {"color": "#C9A84C", "size": 14}},
                    height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F0EAD6", legend={"font": {"color": "#F0EAD6"}, "bgcolor": "rgba(13,17,23,0.8)"},
                    xaxis={"gridcolor": "rgba(255,255,255,0.05)", "title": "Year"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.05)", "title": "Elo Rating"},
                    hovermode="x unified",
                )
                st.plotly_chart(fig_elo, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No Elo history available yet.")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif "About" in page:
        st.markdown('<div class="section-header">ℹ️ About This App</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:2rem;line-height:1.8;color:#F0EAD6;">
        
        <h3 style="color:#C9A84C;font-family:'Bebas Neue',sans-serif;letter-spacing:2px;">FIFA WORLD CUP 2026 AI PREDICTOR</h3>
        
        <p>This platform uses machine learning to predict international football match outcomes and simulate the entire FIFA World Cup 2026 tournament.</p>
        
        <h4 style="color:#C9A84C;">🤖 Models Used</h4>
        <ul style="color:#8B9BB4;">
            <li>Logistic Regression (baseline)</li>
            <li>Random Forest</li>
            <li>Gradient Boosting</li>
            <li>XGBoost (primary)</li>
            <li>CatBoost</li>
            <li>LightGBM</li>
        </ul>
        
        <h4 style="color:#C9A84C;">📊 Features</h4>
        <ul style="color:#8B9BB4;">
            <li>Dynamic Elo ratings (updated match by match)</li>
            <li>Recent form (last 5 matches)</li>
            <li>Win rate, avg goals scored/conceded</li>
            <li>World Cup experience</li>
            <li>Home advantage factor</li>
            <li>Elo difference & ranking gap</li>
        </ul>
        
        <h4 style="color:#C9A84C;">🎲 Monte Carlo Simulation</h4>
        <p>The tournament simulator runs 100–2000 complete World Cup simulations using probabilistic match outcomes weighted by each model's predictions. Championship probabilities converge with more simulations.</p>
        
        <h4 style="color:#C9A84C;">🚀 Quick Start</h4>
        <pre style="background:#050810;border:1px solid rgba(201,168,76,0.2);border-radius:8px;padding:1rem;color:#C9A84C;font-size:0.85rem;">
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
        </pre>
        </div>
        """, unsafe_allow_html=True)
    
    # ── LIVE COUNTDOWN AUTO-REFRESH ───────────────────────────────────────────
    # Refresh every second for the countdown
    # Auto-refresh every second for the countdown timer
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1000, key="clock")
    except ImportError:
        pass  # countdown updates on interaction if package not installed


if __name__ == "__main__":
    main()
