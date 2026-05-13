"""
FIFA ML Business Analytics — Streamlit App
Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA ML Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark green football theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0f0a 0%, #0d1b0d 50%, #081208 100%);
    color: #e8f5e9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2010 0%, #071408 100%);
    border-right: 1px solid #1a3a1a;
}
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }

/* ── Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0d3b12 0%, #1a5c20 50%, #0d3b12 100%);
    border-radius: 16px;
    border: 1px solid #2a7a30;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(76,175,80,0.15) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem; font-weight: 700;
    color: #4caf50;
    text-shadow: 0 0 30px rgba(76,175,80,0.4);
    letter-spacing: 2px;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem; color: #81c784;
    margin-top: 0.5rem; font-weight: 300;
    letter-spacing: 1px;
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #0d2010, #112a13);
    border: 1px solid #2a5a2a;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4caf50; }
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem; font-weight: 700; color: #66bb6a;
}
.metric-label { font-size: 0.8rem; color: #81c784; margin-top: 4px; }

.prediction-box {
    background: linear-gradient(135deg, #0d2010, #133318);
    border: 2px solid #4caf50;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(76,175,80,0.15);
}
.pred-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem; font-weight: 700; color: #69f0ae;
}
.pred-label { font-size: 1rem; color: #a5d6a7; margin-bottom: 0.5rem; }

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem; font-weight: 600;
    color: #4caf50; letter-spacing: 1px;
    border-bottom: 1px solid #1e4a1e;
    padding-bottom: 0.5rem; margin-bottom: 1.2rem;
}

/* ── Sliders & Inputs ── */
.stSlider > div > div > div > div { background: #4caf50 !important; }
.stSlider [data-testid="stThumbValue"] { color: #4caf50 !important; }
.stSelectbox > div > div { 
    background: #0d2010; border-color: #2a5a2a; color: #c8e6c9;
}
.stNumberInput > div > div > input {
    background: #0d2010; border-color: #2a5a2a; color: #c8e6c9;
}
.stTextInput > div > div > input {
    background: #0d2010; border-color: #2a5a2a; color: #c8e6c9;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #388e3c);
    color: white; border: none; border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem; font-weight: 600; letter-spacing: 1px;
    padding: 0.6rem 2rem; width: 100%;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(76,175,80,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #43a047);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(76,175,80,0.4);
}

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
thead tr th { background: #1a3a1a !important; color: #4caf50 !important; }
tbody tr:nth-child(odd) td { background: #0d1f0d !important; }
tbody tr:nth-child(even) td { background: #0a1a0a !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { 
    background: #0d1f0d; border-radius: 8px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #81c784;
    font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1a4a1a; color: #4caf50;
    border-radius: 6px;
}

/* ── Position badge ── */
.badge {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 20px; font-weight: 700;
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;
    letter-spacing: 1px;
}
.badge-att { background: #b71c1c; color: #ffcdd2; }
.badge-mid { background: #1565c0; color: #bbdefb; }
.badge-def { background: #1b5e20; color: #c8e6c9; }
.badge-gk  { background: #e65100; color: #ffe0b2; }

/* ── Divider ── */
hr { border-color: #1a3a1a !important; }

/* ── Alert success ── */
.stSuccess { background: #0d2a0d; border-color: #4caf50; }
.stInfo    { background: #0a1f2a; border-color: #29b6f6; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1f0d; }
::-webkit-scrollbar-thumb { background: #2a5a2a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & META
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_everything():
    with open("models/meta.json") as f:
        meta = json.load(f)

    reg_models, clf_models = {}, {}
    for name, key in meta["reg_model_keys"].items():
        try:
            reg_models[name] = joblib.load(f"models/{key}.pkl")
        except Exception:
            pass
    for name, key in meta["clf_model_keys"].items():
        try:
            clf_models[name] = joblib.load(f"models/{key}.pkl")
        except Exception:
            pass

    scaler_reg = joblib.load("models/scaler_reg.pkl")
    scaler_clf = joblib.load("models/scaler_clf.pkl")
    le         = joblib.load("models/label_encoder.pkl")
    df         = pd.read_csv("data/fifa_synthetic.csv")
    return meta, reg_models, clf_models, scaler_reg, scaler_clf, le, df


try:
    meta, reg_models, clf_models, scaler_reg, scaler_clf, le, df = load_everything()
except FileNotFoundError:
    st.error("⚠️ Models not found! Please run `python train_models.py` first.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
POSITION_BADGE = {
    'Attacker':   ('<span class="badge badge-att">⚡ Attacker</span>', '🔴'),
    'Midfielder': ('<span class="badge badge-mid">🔵 Midfielder</span>', '🔵'),
    'Defender':   ('<span class="badge badge-def">🛡️ Defender</span>', '🟢'),
    'Goalkeeper': ('<span class="badge badge-gk">🧤 Goalkeeper</span>', '🟠'),
}

def fmt_eur(val):
    if val >= 1_000_000:
        return f"€{val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"€{val/1_000:.1f}K"
    return f"€{val:,.0f}"


def predict_value(model_name, features_dict):
    X = pd.DataFrame([features_dict])[meta["reg_features"]]
    if model_name in meta["needs_scale_reg"]:
        X = scaler_reg.transform(X)
        X = pd.DataFrame(X, columns=meta["reg_features"])
    model = reg_models[model_name]
    return float(model.predict(X)[0])


def predict_position(model_name, features_dict):
    X = pd.DataFrame([features_dict])[meta["clf_features"]]
    if model_name in meta["needs_scale_clf"]:
        X_sc = scaler_clf.transform(X)
        X = pd.DataFrame(X_sc, columns=meta["clf_features"])
    model = clf_models[model_name]
    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        proba = {le.inverse_transform([i])[0]: round(float(v)*100,1)
                 for i, v in enumerate(p)}
    return label, proba


def mpl_dark():
    plt.rcParams.update({
        'figure.facecolor':  '#0a0f0a',
        'axes.facecolor':    '#0d1b0d',
        'axes.edgecolor':    '#2a5a2a',
        'axes.labelcolor':   '#c8e6c9',
        'xtick.color':       '#81c784',
        'ytick.color':       '#81c784',
        'text.color':        '#e8f5e9',
        'grid.color':        '#1a3a1a',
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
    })
    return '#69f0ae', '#4caf50', '#1b5e20'


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:3rem'>⚽</div>
        <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;color:#4caf50;'>FIFA ML Analytics</div>
        <div style='font-size:0.75rem;color:#81c784;'>FAST-NUCES · Spring 2026</div>
    </div>
    <hr style='border-color:#1a3a1a;margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠 Home", "💰 Value Prediction", "🎯 Position Classifier",
         "📊 Model Comparison", "📈 Data Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.8rem;color:#81c784;'>
    <b>Dataset:</b> {len(df):,} players<br>
    <b>Features:</b> 15 attributes<br>
    <b>Reg models:</b> {len(reg_models)}<br>
    <b>Clf models:</b> {len(clf_models)}<br>
    <b>Best Reg:</b> {meta['best_reg']}<br>
    <b>Best Clf:</b> {meta['best_clf']}
    </div>
    """, unsafe_allow_html=True)

page = nav.split(" ", 1)[1]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>⚽ FIFA ML ANALYTICS</div>
        <div class='hero-subtitle'>
            Machine Learning for Business Analytics · Spring 2026 · FAST-NUCES
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    reg_res = meta["reg_results"]
    clf_res = meta["clf_results"]
    best_r   = meta["best_reg"]
    best_c   = meta["best_clf"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>🧑‍🤝‍🧑 Players in Dataset</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(reg_models) + len(clf_models)}</div>
            <div class='metric-label'>🤖 ML Models Trained</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{reg_res[best_r]['R2']:.4f}</div>
            <div class='metric-label'>📐 Best Regression R²</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{clf_res[best_c]['F1_Score']:.4f}</div>
            <div class='metric-label'>🎯 Best Classifier F1</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Problem 1 — Regression</div>", unsafe_allow_html=True)
        st.markdown("""
        **Target:** `value_eur` — Player Market Value (€)

        Football clubs spend hundreds of millions every transfer window.
        Overpaying or underpaying for talent has serious financial consequences.
        This model predicts a player's fair market value based on **15 key attributes**
        including skill ratings, age, wage, and international reputation.

        > *"Know your player's worth before the deal is signed."*
        """)

        # Mini regression bar chart
        mpl_dark()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names  = list(reg_res.keys())
        r2vals = [reg_res[n]['R2'] for n in names]
        colors = ['#4caf50' if n == best_r else '#1a4a1a' for n in names]
        bars = ax.barh(names, r2vals, color=colors, edgecolor='#2a6a2a', linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("R² Score", fontsize=9)
        ax.set_title("Regression Model R² Comparison", fontsize=10, color='#4caf50', pad=8)
        ax.axvline(0.9, color='#ff8f00', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(True, axis='x')
        for bar, val in zip(bars, r2vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7, color='#c8e6c9')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Problem 2 — Classification</div>", unsafe_allow_html=True)
        st.markdown("""
        **Target:** `player_positions` — Attacker / Midfielder / Defender / Goalkeeper

        Scouting departments must quickly evaluate thousands of players.
        Manually assigning roles wastes time and introduces bias.
        This model classifies any player into their **optimal position**
        using physical and technical skill attributes.

        > *"Deploy the right player in the right role, every time."*
        """)

        # Mini clf bar chart
        fig, ax = plt.subplots(figsize=(6, 3.5))
        cnames = list(clf_res.keys())
        f1vals = [clf_res[n]['F1_Score'] for n in cnames]
        colors2= ['#4caf50' if n == best_c else '#1a3a5a' for n in cnames]
        bars2  = ax.barh(cnames, f1vals, color=colors2, edgecolor='#2a4a7a', linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Weighted F1 Score", fontsize=9)
        ax.set_title("Classification Model F1 Comparison", fontsize=10, color='#4caf50', pad=8)
        ax.axvline(0.9, color='#ff8f00', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(True, axis='x')
        for bar, val in zip(bars2, f1vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7, color='#c8e6c9')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Report Structure</div>", unsafe_allow_html=True)
    with st.expander("View project submission format"):
        st.markdown("""
        | Section | Content |
        |---|---|
        | **Problem Statement** | Predict player market value (regression) + classify position (classification) |
        | **Objective** | Help football clubs make data-driven transfer & scouting decisions |
        | **Data** | Synthetic FIFA-inspired dataset with 4,000 players, 15 features |
        | **Results** | Model comparison tables, visualizations, best model reasoning |
        | **Code** | Modular Python: `train_models.py` + `app.py` |
        | **Conclusion** | Ensemble methods dominate; business applications discussed |
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: VALUE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Value Prediction":
    st.markdown("<div class='hero-header'><div class='hero-title'>💰 Player Market Value Predictor</div><div class='hero-subtitle'>Enter any player's attributes to predict their fair market value</div></div>", unsafe_allow_html=True)

    col_inp, col_res = st.columns([1.2, 1])

    with col_inp:
        st.markdown("<div class='section-title'>Player Attributes</div>", unsafe_allow_html=True)

        player_name = st.text_input("Player Name (optional)", placeholder="e.g. Kylian Mbappe")

        c1, c2 = st.columns(2)
        with c1:
            age     = st.slider("Age", 16, 40, 24)
            overall = st.slider("Overall Rating", 40, 99, 80)
            pace    = st.slider("Pace", 20, 99, 75)
            shooting= st.slider("Shooting", 20, 99, 70)
            passing = st.slider("Passing", 20, 99, 72)
            dribbling=st.slider("Dribbling", 20, 99, 78)
            defending=st.slider("Defending", 10, 99, 40)
        with c2:
            physicality   = st.slider("Physicality", 20, 99, 70)
            height_cm     = st.slider("Height (cm)", 155, 205, 180)
            weight_kg     = st.slider("Weight (kg)", 55, 105, 76)
            intl_rep      = st.slider("International Reputation ⭐", 1, 5, 2)
            weak_foot     = st.slider("Weak Foot ⭐", 1, 5, 3)
            skill_moves   = st.slider("Skill Moves ⭐", 1, 5, 3)
            wage_eur      = st.number_input("Weekly Wage (€)", 1000, 500000, 50000, step=1000)
            release_clause= st.number_input("Release Clause (€)", 10000, 500000000, 5000000, step=100000)

        reg_model_name = st.selectbox("Model to Use", list(reg_models.keys()),
                                      index=list(reg_models.keys()).index(meta["best_reg"]))
        predict_btn = st.button("🔍 Predict Market Value", key="pred_val")

    with col_res:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        features = {
            'age': age, 'overall': overall, 'pace': pace, 'shooting': shooting,
            'passing': passing, 'dribbling': dribbling, 'defending': defending,
            'physicality': physicality, 'height_cm': height_cm, 'weight_kg': weight_kg,
            'international_reputation': intl_rep, 'weak_foot': weak_foot,
            'skill_moves': skill_moves, 'wage_eur': wage_eur,
            'release_clause_eur': release_clause
        }

        if predict_btn or True:  # Show prediction on load too
            value = predict_value(reg_model_name, features)
            display_name = player_name if player_name else "This Player"

            st.markdown(f"""
            <div class='prediction-box'>
                <div class='pred-label'>💰 Estimated Market Value for <b>{display_name}</b></div>
                <div class='pred-value'>{fmt_eur(value)}</div>
                <div style='color:#81c784;font-size:0.85rem;margin-top:0.5rem;'>
                    Model: {reg_model_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tier classification
            if value >= 80_000_000:
                tier, tier_color = "🌟 World Class / Icon", "#ffd700"
            elif value >= 40_000_000:
                tier, tier_color = "⭐ Elite", "#69f0ae"
            elif value >= 15_000_000:
                tier, tier_color = "💪 Top Player", "#4caf50"
            elif value >= 5_000_000:
                tier, tier_color = "📈 Quality Player", "#81c784"
            else:
                tier, tier_color = "🌱 Developing", "#a5d6a7"

            st.markdown(f"""
            <div style='background:#0d2010;border:1px solid #2a5a2a;border-radius:10px;
                        padding:1rem;text-align:center;'>
                <div style='font-size:0.85rem;color:#81c784;'>Market Tier</div>
                <div style='font-size:1.5rem;font-weight:700;color:{tier_color};
                            font-family:Rajdhani;'>{tier}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Radar chart of attributes
            mpl_dark()
            categories = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']
            vals = [pace, shooting, passing, dribbling, defending, physicality]
            vals += vals[:1]
            angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.set_facecolor('#0d1b0d')
            fig.patch.set_facecolor('#0d1b0d')
            ax.plot(angles, vals, 'o-', linewidth=2, color='#4caf50')
            ax.fill(angles, vals, alpha=0.25, color='#4caf50')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9, color='#c8e6c9')
            ax.set_ylim(0, 99)
            ax.set_yticks([25, 50, 75])
            ax.set_yticklabels(['25','50','75'], size=7, color='#4a6a4a')
            ax.grid(color='#1a3a1a', linestyle='--', linewidth=0.5)
            ax.spines['polar'].set_color('#2a5a2a')
            ax.set_title(f"Skill Profile — {display_name}", pad=15,
                         color='#4caf50', fontsize=10, fontfamily='Rajdhani')
            st.pyplot(fig)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>All Models Comparison</div>", unsafe_allow_html=True)

        all_preds = {}
        for mname in reg_models:
            try:
                all_preds[mname] = predict_value(mname, features)
            except Exception:
                pass

        pred_df = pd.DataFrame({
            "Model": list(all_preds.keys()),
            "Predicted Value": [fmt_eur(v) for v in all_preds.values()],
            "R²": [meta["reg_results"][m]["R2"] for m in all_preds],
        })
        pred_df = pred_df.sort_values("R²", ascending=False).reset_index(drop=True)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: POSITION CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Position Classifier":
    st.markdown("<div class='hero-header'><div class='hero-title'>🎯 Player Position Classifier</div><div class='hero-subtitle'>Predict the optimal playing position from skill attributes</div></div>", unsafe_allow_html=True)

    col_inp, col_res = st.columns([1.2, 1])

    with col_inp:
        st.markdown("<div class='section-title'>Player Attributes</div>", unsafe_allow_html=True)

        player_name2 = st.text_input("Player Name (optional)", placeholder="e.g. Sergio Ramos", key="clf_name")

        c1, c2 = st.columns(2)
        with c1:
            age2       = st.slider("Age", 16, 40, 25, key="c_age")
            overall2   = st.slider("Overall Rating", 40, 99, 78, key="c_ov")
            pace2      = st.slider("Pace", 20, 99, 65, key="c_pa")
            shooting2  = st.slider("Shooting", 20, 99, 45, key="c_sh")
            passing2   = st.slider("Passing", 20, 99, 72, key="c_ps")
            dribbling2 = st.slider("Dribbling", 20, 99, 60, key="c_dr")
        with c2:
            defending2   = st.slider("Defending", 10, 99, 85, key="c_df")
            physicality2 = st.slider("Physicality", 20, 99, 80, key="c_ph")
            height2      = st.slider("Height (cm)", 155, 205, 184, key="c_ht")
            weight2      = st.slider("Weight (kg)", 55, 105, 80, key="c_wt")
            intl2        = st.slider("International Reputation ⭐", 1, 5, 2, key="c_ir")
            weak2        = st.slider("Weak Foot ⭐", 1, 5, 3, key="c_wf")
            skill2       = st.slider("Skill Moves ⭐", 1, 5, 2, key="c_sm")

        clf_model_name = st.selectbox("Model to Use", list(clf_models.keys()),
                                      index=list(clf_models.keys()).index(meta["best_clf"]))
        pred_pos_btn = st.button("🎯 Classify Position", key="pred_pos")

    with col_res:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        clf_features = {
            'age': age2, 'overall': overall2, 'pace': pace2, 'shooting': shooting2,
            'passing': passing2, 'dribbling': dribbling2, 'defending': defending2,
            'physicality': physicality2, 'height_cm': height2, 'weight_kg': weight2,
            'international_reputation': intl2, 'weak_foot': weak2, 'skill_moves': skill2
        }

        pos_label, proba = predict_position(clf_model_name, clf_features)
        badge_html, emoji = POSITION_BADGE.get(pos_label, ('', '❓'))
        display_name2 = player_name2 if player_name2 else "This Player"

        st.markdown(f"""
        <div class='prediction-box'>
            <div class='pred-label'>🎽 Best Position for <b>{display_name2}</b></div>
            <div style='margin:1rem 0;'>{badge_html}</div>
            <div style='font-size:4rem;'>{emoji}</div>
            <div style='color:#81c784;font-size:0.85rem;margin-top:0.5rem;'>
                Model: {clf_model_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if proba:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Confidence Breakdown</div>", unsafe_allow_html=True)

            pos_colors = {'Attacker':'#ef5350','Midfielder':'#42a5f5',
                          'Defender':'#66bb6a','Goalkeeper':'#ff8f00'}
            mpl_dark()
            fig, ax = plt.subplots(figsize=(5, 3))
            ppos   = list(proba.keys())
            pvals  = list(proba.values())
            cols   = [pos_colors.get(p,'#4caf50') for p in ppos]
            bars   = ax.bar(ppos, pvals, color=cols, edgecolor='#2a2a2a', linewidth=0.5,
                            width=0.5)
            ax.set_ylabel("Confidence (%)", fontsize=9)
            ax.set_ylim(0, 110)
            ax.set_title("Position Probabilities", color='#4caf50', fontsize=10,
                         fontfamily='Rajdhani', pad=8)
            ax.grid(True, axis='y')
            for bar, val in zip(bars, pvals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='#e8f5e9')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>All Models — Position Vote</div>", unsafe_allow_html=True)
        vote_data = []
        for mname in clf_models:
            try:
                lbl, pr = predict_position(mname, clf_features)
                conf = f"{pr[lbl]:.1f}%" if pr else "N/A"
                vote_data.append({"Model": mname, "Predicted Position": lbl, "Confidence": conf,
                                  "F1": meta["clf_results"][mname]["F1_Score"]})
            except Exception:
                pass
        vote_df = pd.DataFrame(vote_data).sort_values("F1", ascending=False).reset_index(drop=True)
        st.dataframe(vote_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown("<div class='hero-header'><div class='hero-title'>📊 Model Comparison Dashboard</div><div class='hero-subtitle'>Full performance analysis across all trained models</div></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💰 Regression Models", "🎯 Classification Models"])

    # ── REGRESSION ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-title'>Regression Results — Predicting Player Value (€)</div>", unsafe_allow_html=True)

        reg_df = pd.DataFrame([
            {"Model": n, "RMSE (€)": f"{v['RMSE']:,.0f}",
             "MAE (€)": f"{v['MAE']:,.0f}", "R² Score": v['R2'],
             "Best": "✅" if n == meta['best_reg'] else ""}
            for n, v in meta["reg_results"].items()
        ]).sort_values("R² Score", ascending=False).reset_index(drop=True)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            mpl_dark()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            names  = [n for n in meta["reg_results"]]
            r2s    = [meta["reg_results"][n]["R2"] for n in names]
            colors = ['#4caf50' if n==meta["best_reg"] else '#1a3a2a' for n in names]
            ax.barh(names, r2s, color=colors, edgecolor='#2a6a2a', linewidth=0.5)
            ax.set_xlabel("R² Score")
            ax.set_title("R² Score — All Regression Models", color='#4caf50',
                         fontsize=11, fontfamily='Rajdhani')
            ax.axvline(0.95, color='#ff8f00', linestyle='--', linewidth=1, label='0.95 threshold')
            ax.legend(fontsize=8)
            ax.grid(True, axis='x')
            for i,(val,name) in enumerate(zip(r2s,names)):
                ax.text(val+0.003, i, f'{val:.4f}', va='center', fontsize=7.5, color='#c8e6c9')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            mpl_dark()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            maes = [meta["reg_results"][n]["MAE"] for n in names]
            colors2 = ['#69f0ae' if n==meta["best_reg"] else '#1a4a2a' for n in names]
            ax.barh(names, maes, color=colors2, edgecolor='#2a6a2a', linewidth=0.5)
            ax.set_xlabel("MAE (€)")
            ax.set_title("Mean Absolute Error — All Regression Models", color='#4caf50',
                         fontsize=11, fontfamily='Rajdhani')
            ax.grid(True, axis='x')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{x/1000:.0f}K'))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📝 Business Reasoning</div>", unsafe_allow_html=True)
        st.markdown(f"""
        The **{meta['best_reg']}** model achieves the highest R² of
        **{meta['reg_results'][meta['best_reg']]['R2']:.4f}**, meaning it explains
        {meta['reg_results'][meta['best_reg']]['R2']*100:.1f}% of variance in player market values.

        | Model Category | Insight |
        |---|---|
        | **Linear models** (Linear, Ridge, Lasso) | Strong when features have linear relationships with target; good baseline |
        | **Ensemble methods** (RF, GB, ET) | Capture non-linear interactions between age, rating, and wage |
        | **KNN / SVR** | Effective but computationally heavy; sensitive to scale |
        | **Best choice for business** | {meta['best_reg']} — highest accuracy with interpretable feature importances |

        **Recommendation:** Use {meta['best_reg']} in production transfer decision systems.
        """)

    # ── CLASSIFICATION ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-title'>Classification Results — Predicting Player Position</div>", unsafe_allow_html=True)

        clf_df = pd.DataFrame([
            {"Model": n, "Accuracy": v['Accuracy'], "F1 Score (Weighted)": v['F1_Score'],
             "Best": "✅" if n==meta['best_clf'] else ""}
            for n, v in meta["clf_results"].items()
        ]).sort_values("F1 Score (Weighted)", ascending=False).reset_index(drop=True)
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            mpl_dark()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            cnames = list(meta["clf_results"].keys())
            accs   = [meta["clf_results"][n]["Accuracy"] for n in cnames]
            f1s    = [meta["clf_results"][n]["F1_Score"] for n in cnames]
            x = np.arange(len(cnames))
            w = 0.35
            ax.bar(x - w/2, accs, w, label='Accuracy', color='#4caf50', alpha=0.85)
            ax.bar(x + w/2, f1s,  w, label='F1 Score', color='#29b6f6', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(cnames, rotation=35, ha='right', fontsize=7.5)
            ax.set_ylim(0, 1.1)
            ax.set_title("Accuracy vs F1 Score", color='#4caf50', fontsize=11,
                         fontfamily='Rajdhani')
            ax.legend()
            ax.grid(True, axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            # Scatter: Acc vs F1
            mpl_dark()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for i, n in enumerate(cnames):
                a = meta["clf_results"][n]["Accuracy"]
                f = meta["clf_results"][n]["F1_Score"]
                color = '#4caf50' if n == meta["best_clf"] else '#2a5a4a'
                size  = 200 if n == meta["best_clf"] else 100
                ax.scatter(a, f, s=size, color=color, zorder=3,
                           edgecolors='#69f0ae', linewidths=0.8)
                ax.annotate(n.split()[0], (a, f), textcoords="offset points",
                            xytext=(5,4), fontsize=7, color='#c8e6c9')
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("F1 Score")
            ax.set_title("Accuracy vs F1 — Model Scatter", color='#4caf50',
                         fontsize=11, fontfamily='Rajdhani')
            ax.plot([0.4,1],[0.4,1], '--', color='#1a3a2a', linewidth=0.8)
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📝 Business Reasoning</div>", unsafe_allow_html=True)
        st.markdown(f"""
        The **{meta['best_clf']}** achieves F1 = **{meta['clf_results'][meta['best_clf']]['F1_Score']:.4f}**
        on position classification.

        | Aspect | Detail |
        |---|---|
        | **4 Classes** | Attacker, Midfielder, Defender, Goalkeeper |
        | **Weighted F1** | Accounts for class imbalance (GKs are rare) |
        | **Best for scouts** | {meta['best_clf']} — high precision across all positions |
        | **Business value** | Automated position tagging saves 100s of scouting hours |

        **Recommendation:** Deploy {meta['best_clf']} in the scouting system with a confidence threshold.
        Flagging low-confidence predictions for human review reduces errors.
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("<div class='hero-header'><div class='hero-title'>📈 Data Explorer</div><div class='hero-subtitle'>Explore the FIFA player dataset with interactive visualizations</div></div>", unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df):,}</div><div class='metric-label'>Total Players</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{fmt_eur(df['value_eur'].median())}</div><div class='metric-label'>Median Value</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df['overall'].mean():.1f}</div><div class='metric-label'>Avg Overall Rating</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df['age'].mean():.1f}</div><div class='metric-label'>Avg Age</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📋 Raw Data"])

    with tab1:
        mpl_dark()
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            pos_counts = df['player_positions'].value_counts()
            pos_colors_map = {'Attacker':'#ef5350','Midfielder':'#42a5f5',
                              'Defender':'#66bb6a','Goalkeeper':'#ff8f00'}
            bars = ax.bar(pos_counts.index, pos_counts.values,
                          color=[pos_colors_map.get(p,'#4caf50') for p in pos_counts.index],
                          edgecolor='#2a2a2a', linewidth=0.5)
            ax.set_title("Players by Position", color='#4caf50', fontsize=11,
                         fontfamily='Rajdhani')
            ax.set_ylabel("Count")
            ax.grid(True, axis='y')
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                        str(int(bar.get_height())), ha='center', fontsize=9,
                        color='#e8f5e9')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df['overall'], bins=25, color='#4caf50', edgecolor='#2a2a2a',
                    linewidth=0.5, alpha=0.85)
            ax.set_title("Overall Rating Distribution", color='#4caf50',
                         fontsize=11, fontfamily='Rajdhani')
            ax.set_xlabel("Overall Rating")
            ax.set_ylabel("Frequency")
            ax.grid(True, axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        mpl_dark()
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df['age'], bins=20, color='#29b6f6', edgecolor='#2a2a2a',
                    linewidth=0.5, alpha=0.85)
            ax.set_title("Age Distribution", color='#4caf50', fontsize=11, fontfamily='Rajdhani')
            ax.set_xlabel("Age")
            ax.set_ylabel("Frequency")
            ax.grid(True, axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df['value_eur']/1e6, bins=30, color='#ffd54f', edgecolor='#2a2a2a',
                    linewidth=0.5, alpha=0.85)
            ax.set_title("Market Value Distribution", color='#4caf50',
                         fontsize=11, fontfamily='Rajdhani')
            ax.set_xlabel("Value (€M)")
            ax.set_ylabel("Frequency")
            ax.grid(True, axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Box plot value by position
        mpl_dark()
        fig, ax = plt.subplots(figsize=(10, 4))
        pos_order = ['Attacker','Midfielder','Defender','Goalkeeper']
        pos_data  = [df[df['player_positions']==p]['value_eur'].values/1e6
                     for p in pos_order]
        bp = ax.boxplot(pos_data, labels=pos_order, patch_artist=True,
                        medianprops=dict(color='#ffd54f', linewidth=2))
        box_colors = ['#ef5350','#42a5f5','#66bb6a','#ff8f00']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title("Market Value Distribution by Position", color='#4caf50',
                     fontsize=12, fontfamily='Rajdhani')
        ax.set_ylabel("Value (€M)")
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        mpl_dark()
        num_cols = ['age','overall','pace','shooting','passing','dribbling',
                    'defending','physicality','wage_eur','value_eur']
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax, annot=True, fmt='.2f',
                    annot_kws={'size':8}, linewidths=0.5, linecolor='#0d1b0d',
                    vmin=-1, vmax=1, cbar_kws={'shrink':0.8})
        ax.set_title("Feature Correlation Heatmap", color='#4caf50',
                     fontsize=13, fontfamily='Rajdhani', pad=15)
        ax.tick_params(colors='#c8e6c9', labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Top Correlations with Market Value</div>", unsafe_allow_html=True)
        corr_val = corr['value_eur'].drop('value_eur').sort_values(ascending=False)
        corr_df = pd.DataFrame({
            "Feature": corr_val.index,
            "Correlation with Value": corr_val.values.round(4)
        })
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("<div class='section-title'>Dataset Sample</div>", unsafe_allow_html=True)
        st.markdown(f"**{len(df):,} players** · **{len(df.columns)}** features · Synthetic FIFA-inspired data")

        filter_pos = st.multiselect("Filter by Position",
                                    df['player_positions'].unique().tolist(),
                                    default=df['player_positions'].unique().tolist())
        filtered = df[df['player_positions'].isin(filter_pos)]

        display_df = filtered.copy()
        display_df['value_eur'] = display_df['value_eur'].apply(fmt_eur)
        display_df['wage_eur']  = display_df['wage_eur'].apply(lambda x: f"€{x:,}")
        st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 100 of {len(filtered):,} filtered players")
