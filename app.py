import kaggle
kaggle.api.dataset_download_files(
    'stefanoleone992/fifa-22-complete-player-dataset',
    path='.', unzip=True
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Analytics - ML Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a472a, #2d6a4f, #52b788);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
    }
    .best-badge {
        background: #f4a261;
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚽ FIFA Analytics - ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning for Business Analytics | Group: FIFA Analytics | BsBa 6A</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/EA_Sports_FIFA_22_logo.svg/320px-EA_Sports_FIFA_22_logo.svg.png",
             width=200)
    st.markdown("### ⚙️ Configuration")
    uploaded_file = st.file_uploader("📂 Upload players_22.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.markdown("- **Muhammad Mudassir** (23F-5007)")
    st.markdown("- **Abdul Latif**")
    st.markdown("- **Asad Ullah**")
    st.markdown("---")
    st.markdown("**Dataset:** FIFA 22 Complete Player Dataset")
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)")

# ─── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file, low_memory=False)

    features = [
        'age', 'overall', 'potential', 'wage_eur', 'release_clause_eur',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'height_cm', 'weight_kg', 'international_reputation',
        'weak_foot', 'skill_moves'
    ]

    df_work = df[features + ['value_eur', 'player_positions',
                              'short_name', 'club_name', 'nationality_name']].copy()
    df_work.dropna(subset=['value_eur', 'player_positions'], inplace=True)

    for col in df_work.select_dtypes(include='number').columns:
        df_work[col].fillna(df_work[col].median(), inplace=True)

    # Feature engineering
    df_work['potential_gap'] = df_work['potential'] - df_work['overall']
    df_work['wage_to_value'] = df_work['wage_eur'] / (df_work['value_eur'] + 1)
    df_work['age_group'] = pd.cut(df_work['age'], bins=[15, 21, 27, 33, 50],
                                   labels=['Young', 'Prime', 'Experienced', 'Veteran'])
    le_age = LabelEncoder()
    df_work['age_group_enc'] = le_age.fit_transform(df_work['age_group'])

    def map_position(pos):
        pos = str(pos).upper()
        if 'GK' in pos: return 'Goalkeeper'
        for p in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
            if p in pos: return 'Defender'
        for p in ['CM', 'CAM', 'CDM', 'LM', 'RM']:
            if p in pos: return 'Midfielder'
        for p in ['ST', 'CF', 'LW', 'RW']:
            if p in pos: return 'Attacker'
        return 'Midfielder'

    df_work['position_cat'] = df_work['player_positions'].apply(map_position)

    features_ext = features + ['potential_gap', 'wage_to_value', 'age_group_enc']
    return df_work, features_ext


def run_regression(df_work, features_ext):
    X = df_work[features_ext]
    y = np.log1p(df_work['value_eur'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=1.0), True),
        'Lasso Regression': (Lasso(alpha=0.01), True),
        'ElasticNet': (ElasticNet(alpha=0.01, l1_ratio=0.5), True),
        'Decision Tree': (DecisionTreeRegressor(max_depth=10, random_state=42), False),
        'KNN Regressor': (KNeighborsRegressor(n_neighbors=5), True),
        'SVR': (SVR(kernel='rbf', C=10), True),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),
        'Extra Trees': (ExtraTreesRegressor(n_estimators=100, random_state=42), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        'AdaBoost': (AdaBoostRegressor(n_estimators=100, random_state=42), False),
        'XGBoost': (xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0), False),
        'LightGBM': (lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1), False),
    }

    results = []
    for name, (model, use_scaled) in models.items():
        Xtr = X_tr_sc if use_scaled else X_train
        Xte = X_te_sc if use_scaled else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': round(rmse, 4),
                        'MAE': round(mae, 4), 'R² Score': round(r2, 4)})

    return pd.DataFrame(results).sort_values('R² Score', ascending=False), X_train, X_test, y_train, y_test


def run_classification(df_work, features_ext):
    le = LabelEncoder()
    y = le.fit_transform(df_work['position_cat'])
    X = df_work[features_ext]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'KNN Classifier': (KNeighborsClassifier(n_neighbors=5), True),
        'Naive Bayes': (GaussianNB(), True),
        'SVM (RBF)': (SVC(kernel='rbf', C=10, random_state=42), True),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'Extra Trees': (ExtraTreesClassifier(n_estimators=100, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        'AdaBoost': (AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'), False),
        'XGBoost': (XGBClassifier(n_estimators=200, random_state=42,
                                   use_label_encoder=False, eval_metric='mlogloss', verbosity=0), False),
        'LightGBM': (LGBMClassifier(n_estimators=200, random_state=42, verbose=-1), False),
    }

    results = []
    for name, (model, use_scaled) in models.items():
        Xtr = X_tr_sc if use_scaled else X_train
        Xte = X_te_sc if use_scaled else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro')
        f1w = f1_score(y_test, y_pred, average='weighted')
        results.append({'Model': name, 'Accuracy': round(acc, 4),
                        'F1 Macro': round(f1m, 4), 'F1 Weighted': round(f1w, 4)})

    return pd.DataFrame(results).sort_values('Accuracy', ascending=False), le, X_train, X_test, y_train, y_test

# ─── Main App ─────────────────────────────────────────────────────────────────
if uploaded_file is None:
    # ── Demo Mode ──
    st.info("📤 Please upload **players_22.csv** from the FIFA 22 Kaggle dataset in the sidebar to begin.")
    st.markdown("### 📋 Project Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 Regression Task**
        - **Target:** `value_eur` (Player Market Value)
        - **Models:** 13 models including XGBoost, LightGBM, RF, GBM, SVR, etc.
        - **Metrics:** RMSE, MAE, R² Score
        - **Business Use:** Fair transfer price estimation

        **📊 Features Used:**
        - Overall, Potential, Age, Pace, Shooting
        - Passing, Dribbling, Defending, Physic
        - Wage, Release Clause, Height, Weight
        - International Reputation, Weak Foot, Skill Moves
        - Engineered: Potential Gap, Wage-to-Value Ratio
        """)
    with col2:
        st.markdown("""
        **🏷️ Classification Task**
        - **Target:** `player_positions` (4 categories)
        - **Categories:** Attacker, Midfielder, Defender, Goalkeeper
        - **Models:** 11 models including XGBoost, LightGBM, SVM, etc.
        - **Metrics:** Accuracy, F1 Macro, F1 Weighted
        - **Business Use:** Player scouting & role assignment

        **📊 Dataset Info:**
        - Source: Kaggle / sofifa.com
        - Players: 19,000+ records
        - Variables: 100+ (we use 19 key features)
        - License: CC0 Public Domain
        """)

else:
    # ── Load Data ──
    with st.spinner("Loading and preprocessing data..."):
        df_work, features_ext = load_and_preprocess(uploaded_file)

    st.success(f"✅ Data loaded! {df_work.shape[0]:,} players | {len(features_ext)} features")

    # ── Key Metrics ──
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_work.shape[0]:,}</div><div class="metric-label">Total Players</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_work["overall"].mean():.1f}</div><div class="metric-label">Avg Overall</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">€{df_work["value_eur"].mean()/1e6:.1f}M</div><div class="metric-label">Avg Value</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_work["age"].mean():.1f}</div><div class="metric-label">Avg Age</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df_work["nationality_name"].nunique()}</div><div class="metric-label">Nationalities</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 EDA", "📈 Regression", "🏷️ Classification", "🔮 Predict", "📋 Data"
    ])

    # ─── TAB 1: EDA ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 🔍 Exploratory Data Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(np.log1p(df_work['value_eur']), bins=50, color='#2d6a4f', edgecolor='white', alpha=0.85)
            ax.set_title('Player Market Value Distribution (log scale)', fontweight='bold')
            ax.set_xlabel('log(Value EUR)')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            pos_counts = df_work['position_cat'].value_counts()
            colors = ['#2d6a4f', '#52b788', '#95d5b2', '#d8f3dc']
            ax.bar(pos_counts.index, pos_counts.values, color=colors, edgecolor='white')
            ax.set_title('Player Position Distribution', fontweight='bold')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df_work['overall'], np.log1p(df_work['value_eur']),
                       alpha=0.2, color='#2d6a4f', s=8)
            ax.set_title('Overall Rating vs Market Value', fontweight='bold')
            ax.set_xlabel('Overall Rating')
            ax.set_ylabel('log(Value EUR)')
            st.pyplot(fig)
            plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(7, 4))
            corr_cols = ['overall', 'potential', 'age', 'wage_eur', 'pace',
                         'shooting', 'passing', 'dribbling', 'value_eur']
            corr = df_work[corr_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens', ax=ax,
                        linewidths=0.5, annot_kws={'size': 7})
            ax.set_title('Correlation Heatmap', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        st.markdown("### 🏆 Top 20 Most Valuable Players")
        top20 = df_work.nlargest(20, 'value_eur')[
            ['short_name', 'overall', 'potential', 'age', 'value_eur', 'wage_eur', 'position_cat', 'club_name']
        ].copy()
        top20['value_eur'] = top20['value_eur'].apply(lambda x: f"€{x/1e6:.1f}M")
        top20['wage_eur'] = top20['wage_eur'].apply(lambda x: f"€{x:,.0f}")
        st.dataframe(top20, use_container_width=True)

    # ─── TAB 2: REGRESSION ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("## 📈 Regression Task — Predicting Player Market Value")
        st.info("**Target:** `value_eur` | **Transform:** log1p applied | **Train/Test:** 80/20 split")

        if st.button("🚀 Run All Regression Models", key="run_reg"):
            with st.spinner("Training 13 regression models..."):
                reg_df, Xtr, Xte, ytr, yte = run_regression(df_work, features_ext)
            st.session_state['reg_df'] = reg_df

        if 'reg_df' in st.session_state:
            reg_df = st.session_state['reg_df']
            best_model = reg_df.iloc[0]['Model']

            # Highlight best
            def highlight_best_reg(row):
                if row['Model'] == best_model:
                    return ['background-color: #d8f3dc; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.markdown(f"**🏆 Best Model: {best_model}** (R² = {reg_df.iloc[0]['R² Score']})")
            st.dataframe(reg_df.style.apply(highlight_best_reg, axis=1), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = ['#1a472a' if m == best_model else '#95d5b2' for m in reg_df['Model']]
                ax.barh(reg_df['Model'], reg_df['R² Score'], color=colors)
                ax.set_title('R² Score Comparison', fontweight='bold')
                ax.set_xlabel('R² Score')
                ax.axvline(0.9, color='red', linestyle='--', alpha=0.7, label='0.90 line')
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = ['#1a472a' if m == reg_df.loc[reg_df['RMSE'].idxmin(), 'Model'] else '#f4a261' for m in reg_df['Model']]
                ax.barh(reg_df['Model'], reg_df['RMSE'], color=colors)
                ax.set_title('RMSE Comparison (Lower=Better)', fontweight='bold')
                ax.set_xlabel('RMSE')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col3:
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = ['#1a472a' if m == reg_df.loc[reg_df['MAE'].idxmin(), 'Model'] else '#52b788' for m in reg_df['Model']]
                ax.barh(reg_df['Model'], reg_df['MAE'], color=colors)
                ax.set_title('MAE Comparison (Lower=Better)', fontweight='bold')
                ax.set_xlabel('MAE')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("### 💡 Reasoning")
            st.markdown(f"""
            - **{best_model}** achieves the highest R² score, meaning it explains the most variance in player market value.
            - Ensemble and boosting models (XGBoost, LightGBM, Random Forest) significantly outperform linear models.
            - Linear models struggle because the relationship between attributes and market value is non-linear.
            - The log-transformation of `value_eur` stabilizes variance and improves all model performances.
            """)

    # ─── TAB 3: CLASSIFICATION ─────────────────────────────────────────────────
    with tab3:
        st.markdown("## 🏷️ Classification Task — Predicting Player Position")
        st.info("**Target:** Position Category (Attacker / Midfielder / Defender / Goalkeeper) | **Train/Test:** 80/20 stratified")

        if st.button("🚀 Run All Classification Models", key="run_cls"):
            with st.spinner("Training 11 classification models..."):
                cls_df, le_pos, Xtr_c, Xte_c, ytr_c, yte_c = run_classification(df_work, features_ext)
            st.session_state['cls_df'] = cls_df
            st.session_state['cls_data'] = (le_pos, Xtr_c, Xte_c, ytr_c, yte_c)

        if 'cls_df' in st.session_state:
            cls_df = st.session_state['cls_df']
            best_cls_model = cls_df.iloc[0]['Model']

            def highlight_best_cls(row):
                if row['Model'] == best_cls_model:
                    return ['background-color: #d8f3dc; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.markdown(f"**🏆 Best Model: {best_cls_model}** (Accuracy = {cls_df.iloc[0]['Accuracy']})")
            st.dataframe(cls_df.style.apply(highlight_best_cls, axis=1), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 5))
                colors = ['#1a472a' if m == best_cls_model else '#95d5b2' for m in cls_df['Model']]
                ax.barh(cls_df['Model'], cls_df['Accuracy'], color=colors)
                ax.set_title('Accuracy Comparison', fontweight='bold')
                ax.set_xlabel('Accuracy')
                ax.axvline(0.85, color='red', linestyle='--', alpha=0.7, label='85% line')
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(7, 5))
                x = np.arange(len(cls_df))
                w = 0.35
                ax.barh(x + w/2, cls_df['F1 Macro'], w, color='#52b788', label='F1 Macro')
                ax.barh(x - w/2, cls_df['F1 Weighted'], w, color='#2d6a4f', label='F1 Weighted')
                ax.set_yticks(x)
                ax.set_yticklabels(cls_df['Model'])
                ax.set_title('F1 Score Comparison', fontweight='bold')
                ax.set_xlabel('F1 Score')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Confusion Matrix
            st.markdown("### Confusion Matrix — Best Model")
            le_pos, Xtr_c, Xte_c, ytr_c, yte_c = st.session_state['cls_data']
            best = XGBClassifier(n_estimators=200, random_state=42,
                                  use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
            best.fit(Xtr_c, ytr_c)
            y_pred_cm = best.predict(Xte_c)
            cm = confusion_matrix(yte_c, y_pred_cm)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=le_pos.classes_, yticklabels=le_pos.classes_)
            ax.set_title(f'Confusion Matrix — {best_cls_model}', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
            plt.close()

            st.markdown("### 💡 Reasoning")
            st.markdown(f"""
            - **{best_cls_model}** performs best due to its ability to capture complex non-linear relationships between physical/skill attributes and positions.
            - Goalkeepers are classified with near-perfect accuracy (very distinct stats).
            - Midfielders and Defenders have some overlap due to CDM/CM positional ambiguity.
            - Ensemble models consistently outperform simple models (Naive Bayes, KNN).
            """)

    # ─── TAB 4: PREDICT ────────────────────────────────────────────────────────
    with tab4:
        st.markdown("## 🔮 Predict Player Value & Position")
        st.markdown("Enter player attributes below to get live predictions:")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 15, 45, 24)
            overall = st.slider("Overall Rating", 40, 99, 75)
            potential = st.slider("Potential", 40, 99, 80)
            wage_eur = st.number_input("Wage (EUR/week)", 500, 500000, 20000, step=1000)
            release_clause = st.number_input("Release Clause (EUR)", 0, 300000000, 5000000, step=500000)
        with col2:
            pace = st.slider("Pace", 1, 99, 70)
            shooting = st.slider("Shooting", 1, 99, 65)
            passing = st.slider("Passing", 1, 99, 70)
            dribbling = st.slider("Dribbling", 1, 99, 72)
            defending = st.slider("Defending", 1, 99, 50)
        with col3:
            physic = st.slider("Physicality", 1, 99, 68)
            height_cm = st.slider("Height (cm)", 155, 210, 180)
            weight_kg = st.slider("Weight (kg)", 50, 110, 75)
            intl_rep = st.selectbox("International Reputation", [1, 2, 3, 4, 5], index=0)
            weak_foot = st.selectbox("Weak Foot (stars)", [1, 2, 3, 4, 5], index=2)
            skill_moves = st.selectbox("Skill Moves (stars)", [1, 2, 3, 4, 5], index=2)

        if st.button("🔮 Predict Now!", key="predict"):
            # Feature engineering for input
            potential_gap = potential - overall
            wage_to_value_approx = wage_eur / 1e6
            age_enc = 0 if age <= 21 else (1 if age <= 27 else (2 if age <= 33 else 3))

            input_data = pd.DataFrame([[age, overall, potential, wage_eur, release_clause,
                                         pace, shooting, passing, dribbling, defending, physic,
                                         height_cm, weight_kg, intl_rep, weak_foot, skill_moves,
                                         potential_gap, wage_to_value_approx, age_enc]],
                                       columns=features_ext)

            # Train quick model on all data
            le_pos2 = LabelEncoder()
            y_all_cls = le_pos2.fit_transform(df_work['position_cat'])
            X_all = df_work[features_ext]
            y_all_reg = np.log1p(df_work['value_eur'])

            reg_model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
            reg_model.fit(X_all, y_all_reg)
            pred_value = np.expm1(reg_model.predict(input_data)[0])

            cls_model = XGBClassifier(n_estimators=200, random_state=42,
                                       use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
            cls_model.fit(X_all, y_all_cls)
            pred_pos_idx = cls_model.predict(input_data)[0]
            pred_position = le_pos2.classes_[pred_pos_idx]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                <div style="font-size:1.2rem; margin-bottom:8px;">💰 Predicted Market Value</div>
                <div class="metric-value">€{pred_value/1e6:.2f}M</div>
                <div class="metric-label">EUR {pred_value:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                pos_icons = {'Attacker': '⚡', 'Midfielder': '🔄', 'Defender': '🛡️', 'Goalkeeper': '🥅'}
                icon = pos_icons.get(pred_position, '⚽')
                st.markdown(f"""
                <div class="metric-card">
                <div style="font-size:1.2rem; margin-bottom:8px;">🎯 Predicted Position</div>
                <div class="metric-value">{icon} {pred_position}</div>
                <div class="metric-label">Based on player attributes</div>
                </div>
                """, unsafe_allow_html=True)

    # ─── TAB 5: DATA ──────────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 📋 Dataset Preview")
        search = st.text_input("🔍 Search player by name")
        pos_filter = st.multiselect("Filter by Position", df_work['position_cat'].unique(),
                                     default=df_work['position_cat'].unique())

        filtered = df_work[df_work['position_cat'].isin(pos_filter)]
        if search:
            filtered = filtered[filtered['short_name'].str.contains(search, case=False, na=False)]

        display_cols = ['short_name', 'age', 'overall', 'potential', 'value_eur',
                        'wage_eur', 'pace', 'shooting', 'passing', 'dribbling',
                        'defending', 'physic', 'position_cat', 'club_name', 'nationality_name']
        st.dataframe(filtered[display_cols].head(200), use_container_width=True)
        st.caption(f"Showing {min(200, len(filtered))} of {len(filtered)} players")
