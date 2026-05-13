"""
FIFA Player ML Project - Model Training Script
Trains regression (value_eur) and classification (player_positions) models
Run this ONCE before launching the Streamlit app.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score
)

# ── Regression Models ─────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ── Classification Models ─────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC FIFA DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  FIFA ML Project — Training All Models")
print("=" * 60)

np.random.seed(42)
N = 4000

ages                   = np.random.randint(16, 38, N)
overall                = np.random.randint(50, 95, N)
pace                   = np.random.randint(30, 99, N)
shooting               = np.random.randint(20, 99, N)
passing                = np.random.randint(20, 99, N)
dribbling              = np.random.randint(20, 99, N)
defending              = np.random.randint(10, 95, N)
physicality            = np.random.randint(30, 99, N)
height_cm              = np.random.randint(160, 200, N)
weight_kg              = np.random.randint(60, 100, N)
international_rep      = np.random.randint(1, 6, N)
weak_foot              = np.random.randint(1, 6, N)
skill_moves            = np.random.randint(1, 6, N)
wage_eur               = (overall ** 2 * 10 + np.random.randint(1000, 50000, N)).astype(int)
release_clause_eur     = (wage_eur * np.random.uniform(30, 150, N)).astype(int)

# Market value formula (realistic, EUR)
value_eur = (
    overall * 80000
    + np.clip(32 - ages, 0, 16) * 25000
    + pace * 5000
    + shooting * 3000
    + dribbling * 6000
    + passing * 2500
    + international_rep * 800000
    + skill_moves * 100000
    + wage_eur * 60
    + np.random.normal(0, 300000, N)
).clip(50000, 200_000_000).astype(int)


# Position derived deterministically from skill attributes (classifiable)
positions = []
for i in range(N):
    s = shooting[i]; p = passing[i]; d = defending[i]
    pa = pace[i]; dr = dribbling[i]; h = height_cm[i]; ph = physicality[i]

    att = s * 0.45 + dr * 0.30 + pa * 0.20 + p * 0.05
    mid = p * 0.40 + dr * 0.25 + pa * 0.20 + s * 0.15
    dfn = d * 0.50 + ph  * 0.25 + h * 0.15  + (99 - s) * 0.10
    gk  = h * 0.50 + ph  * 0.30 + d * 0.15  + (99 - pa) * 0.05

    # Goalkeepers need very high height
    if h < 183:
        gk *= 0.3

    best_pos = max({'Attacker': att, 'Midfielder': mid,
                    'Defender': dfn, 'Goalkeeper': gk}, key=lambda k: {'Attacker': att, 'Midfielder': mid, 'Defender': dfn, 'Goalkeeper': gk}[k])
    positions.append(best_pos)

df = pd.DataFrame({
    'age': ages, 'overall': overall, 'pace': pace, 'shooting': shooting,
    'passing': passing, 'dribbling': dribbling, 'defending': defending,
    'physicality': physicality, 'height_cm': height_cm, 'weight_kg': weight_kg,
    'international_reputation': international_rep,
    'weak_foot': weak_foot, 'skill_moves': skill_moves,
    'wage_eur': wage_eur, 'release_clause_eur': release_clause_eur,
    'value_eur': value_eur,
    'player_positions': positions
})

df.to_csv("data/fifa_synthetic.csv", index=False)
print(f"\n✔ Dataset created: {N} players")
print(df['player_positions'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
REG_FEATURES = [
    'age', 'overall', 'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physicality', 'height_cm', 'weight_kg',
    'international_reputation', 'weak_foot', 'skill_moves',
    'wage_eur', 'release_clause_eur'
]
CLF_FEATURES = [
    'age', 'overall', 'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physicality', 'height_cm', 'weight_kg',
    'international_reputation', 'weak_foot', 'skill_moves'
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  REGRESSION — Predicting Player Market Value (EUR)")
print("=" * 60)

X_reg = df[REG_FEATURES]
y_reg = df['value_eur']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_tr_r_sc  = scaler_reg.fit_transform(X_tr_r)
X_te_r_sc  = scaler_reg.transform(X_te_r)

REG_MODELS = {
    "Linear Regression":        LinearRegression(),
    "Ridge Regression":         Ridge(alpha=1.0),
    "Lasso Regression":         Lasso(alpha=100),
    "ElasticNet":               ElasticNet(alpha=100, l1_ratio=0.5),
    "Decision Tree":            DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees":              ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost":                 AdaBoostRegressor(n_estimators=100, random_state=42),
    "Bagging Regressor":        BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "KNN Regressor":            KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    "SVR":                      SVR(kernel='rbf', C=100000, epsilon=50000),
}

reg_results = {}
best_reg_r2   = -999
best_reg_name = None
best_reg_model = None

for name, model in REG_MODELS.items():
    # Use scaled data for linear/SVM/KNN
    if name in ["Linear Regression","Ridge Regression","Lasso Regression",
                "ElasticNet","KNN Regressor","SVR"]:
        model.fit(X_tr_r_sc, y_tr_r)
        preds = model.predict(X_te_r_sc)
    else:
        model.fit(X_tr_r, y_tr_r)
        preds = model.predict(X_te_r)

    rmse = np.sqrt(mean_squared_error(y_te_r, preds))
    mae  = mean_absolute_error(y_te_r, preds)
    r2   = r2_score(y_te_r, preds)
    reg_results[name] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
    print(f"  {name:<28}  RMSE={rmse:>14,.0f}  MAE={mae:>14,.0f}  R²={r2:.4f}")

    if r2 > best_reg_r2:
        best_reg_r2   = r2
        best_reg_name = name
        best_reg_model = model

print(f"\n✔ Best Regression Model: {best_reg_name} (R²={best_reg_r2:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASSIFICATION — Predicting Player Position")
print("=" * 60)

le = LabelEncoder()
X_clf = df[CLF_FEATURES]
y_clf = le.fit_transform(df['player_positions'])

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

scaler_clf  = StandardScaler()
X_tr_c_sc   = scaler_clf.fit_transform(X_tr_c)
X_te_c_sc   = scaler_clf.transform(X_te_c)

CLF_MODELS = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":            DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees":              ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":                 AdaBoostClassifier(n_estimators=100, random_state=42),
    "Bagging Classifier":       BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "KNN Classifier":           KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "SVM Classifier":           SVC(kernel='rbf', C=10, random_state=42),
    "Naive Bayes":              GaussianNB(),
}

clf_results = {}
best_clf_f1   = -1
best_clf_name = None
best_clf_model = None

for name, model in CLF_MODELS.items():
    needs_scale = name in ["Logistic Regression","KNN Classifier","SVM Classifier"]
    if needs_scale:
        model.fit(X_tr_c_sc, y_tr_c)
        preds = model.predict(X_te_c_sc)
    else:
        model.fit(X_tr_c, y_tr_c)
        preds = model.predict(X_te_c)

    acc = accuracy_score(y_te_c, preds)
    f1  = f1_score(y_te_c, preds, average='weighted')
    clf_results[name] = {"Accuracy": round(acc, 4), "F1_Score": round(f1, 4)}
    print(f"  {name:<28}  Acc={acc:.4f}  F1={f1:.4f}")

    if f1 > best_clf_f1:
        best_clf_f1   = f1
        best_clf_name = name
        best_clf_model = model

print(f"\n✔ Best Classification Model: {best_clf_name} (F1={best_clf_f1:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Saving Models & Artifacts …")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# Save ALL regression models
for name, model in REG_MODELS.items():
    safe = name.replace(" ", "_").lower()
    joblib.dump(model, f"models/reg_{safe}.pkl")

# Save ALL classification models
for name, model in CLF_MODELS.items():
    safe = name.replace(" ", "_").lower()
    joblib.dump(model, f"models/clf_{safe}.pkl")

# Save preprocessors
joblib.dump(scaler_reg, "models/scaler_reg.pkl")
joblib.dump(scaler_clf, "models/scaler_clf.pkl")
joblib.dump(le,         "models/label_encoder.pkl")

# Save metadata
meta = {
    "reg_features":   REG_FEATURES,
    "clf_features":   CLF_FEATURES,
    "classes":        list(le.classes_),
    "best_reg":       best_reg_name,
    "best_clf":       best_clf_name,
    "reg_results":    reg_results,
    "clf_results":    clf_results,
    "reg_model_keys": {n: "reg_" + n.replace(" ","_").lower() for n in REG_MODELS},
    "clf_model_keys": {n: "clf_" + n.replace(" ","_").lower() for n in CLF_MODELS},
    "needs_scale_reg": ["Linear Regression","Ridge Regression","Lasso Regression",
                        "ElasticNet","KNN Regressor","SVR"],
    "needs_scale_clf": ["Logistic Regression","KNN Classifier","SVM Classifier"],
}

with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ All models saved to  models/")
print("✅ Metadata saved to    models/meta.json")
print("\nNow run:  streamlit run app.py")
