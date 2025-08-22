# app.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from streamlit.components.v1 import html
from streamlit import markdown
import h3
from catboost import CatBoostClassifier
from pathlib import Path

st.set_page_config(page_title="Injury Score Only", layout="wide")

# ============================================================
# Robust path resolution (works if app lives in a subfolder)
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

def _repo_root(start: Path) -> Path:
    """Walk up until we find a likely repo root (.git / pyproject / requirements), else filesystem root."""
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    # drive root on Windows, '/' on Unix
    return Path(start.anchor) if start.anchor else start

REPO_ROOT = _repo_root(APP_DIR)

def find_file(filename: str, preferred_subdir: str | None = None) -> Path | None:
    """Try several locations; fall back to a repo-wide search."""
    candidates = []
    # 1) Next to the app
    candidates.append(APP_DIR / filename)
    if preferred_subdir:
        candidates.append(APP_DIR / preferred_subdir / filename)
    # 2) Common bases
    for base in {CWD, APP_DIR.parent, REPO_ROOT}:
        candidates.append(base / filename)
        if preferred_subdir:
            candidates.append(base / preferred_subdir / filename)
    for c in candidates:
        if c.exists():
            return c
    # 3) Repo-wide search
    try:
        return next(REPO_ROOT.rglob(filename))
    except StopIteration:
        return None

# If your assets are under a fixed subfolder (e.g. "assets"), set it here.
ASSETS_SUBDIR = None  # e.g. "assets"

MODEL_PATH  = find_file("catboost_model.pkl",  ASSETS_SUBDIR)
DATA_PATH   = find_file("processed_data.csv",  ASSETS_SUBDIR)
KEPLER_PATH = find_file("kepler_map.html",     ASSETS_SUBDIR)
FOLIUM_PATH = find_file("my_folium_map.html",  ASSETS_SUBDIR)

# Show what we resolved (handy on Streamlit Cloud)
with st.expander("Debug: paths"):
    st.write("APP_DIR:", str(APP_DIR))
    st.write("CWD:", str(CWD))
    st.write("REPO_ROOT:", str(REPO_ROOT))
    st.write("MODEL_PATH:", str(MODEL_PATH) if MODEL_PATH else "❌ not found")
    st.write("DATA_PATH:", str(DATA_PATH) if DATA_PATH else "❌ not found")
    st.write("KEPLER_PATH:", str(KEPLER_PATH) if KEPLER_PATH else "❌ not found")
    st.write("FOLIUM_PATH:", str(FOLIUM_PATH) if FOLIUM_PATH else "❌ not found")

def _need(path: Path | None, label: str):
    if path is None:
        st.error(
            f"❌ Missing required file **{label}** (`{label}`) anywhere under `{REPO_ROOT}`.\n"
            f"Place it next to `app.py` or set `ASSETS_SUBDIR` if it lives in a subfolder."
        )
        st.stop()

# Required: model & data. Maps are optional.
_need(MODEL_PATH, "catboost_model.pkl")
_need(DATA_PATH, "processed_data.csv")

if KEPLER_PATH is None:
    st.warning("Kepler map not found; that panel will be hidden.")
if FOLIUM_PATH is None:
    st.warning("Folium map not found; that panel will be hidden.")

# === Top header: title (left) + score (right) ===
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.title("🚑 Injury Prediction")

with hdr_right:
    score_top = st.empty()      # upper-right metric
    caption_top = st.empty()    # small caption under the metric

# --- Load trained model & data ---
with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)

df = pd.read_csv(DATA_PATH, index_col=0).drop(columns="Injury Severity", errors="ignore")

# ---- Feature schema (your list) ----
feature_names = [
 'Agency Name','ACRS Report Type','Route Type','Collision Type','Weather',
 'Surface Condition','Light','Traffic Control','Driver Substance Abuse',
 'Driver At Fault','Driver Distracted By','Drivers License State',
 'Vehicle Damage Extent','Vehicle First Impact Location','Vehicle Body Type',
 'Vehicle Movement','Vehicle Going Dir','Speed Limit','Vehicle Year',
 'Vehicle Make','Crash Hour','Crash Day','Crash Month','h3_index'
]

# Features to treat as numeric; others are categorical/text
numeric_features = {
    'Speed Limit','Vehicle Year','Crash Hour','Crash Day','Crash Month','Latitude','Longitude'
}
# If your h3_index is numeric, uncomment the next line:
# numeric_features.add('h3_index')

# Define constraints for numeric features
numeric_constraints = {
    'Speed Limit': {'min': 0, 'max': 200, 'step': 5, 'default': 25},
    'Vehicle Year': {'min': 1980, 'max': 2025, 'step': 1, 'default': 2020},
    'Crash Hour': {'min': 0, 'max': 23, 'step': 1, 'default': 12},
    'Crash Day': {'min': 1, 'max': 31, 'step': 1, 'default': 15},
    'Crash Month': {'min': 1, 'max': 12, 'step': 1, 'default': 6},
    'Latitude': {'min': -90.0, 'max': 90.0, 'step': 0.000001, 'default': 39.2904},
    'Longitude': {'min': -180.0, 'max': 180.0, 'step': 0.000001, 'default': -76.6122}
}

# ---- Helpers ----
def coerce_value(name: str, val: str):
    """Convert input string to the right type or None."""
    val = str(val).strip()
    if val == "":
        return None  # let CatBoost handle missing
    if name in numeric_features:
        try:
            # Prefer int when appropriate; else float
            return int(val) if (val.lstrip("-").isdigit()) else float(val)
        except ValueError:
            return None
    return val  # categorical/text

# add categorical option using unique values from the dataset
categorical_options = {}
for feat in feature_names:
    if feat not in numeric_features and feat in df.columns:
        unique_values = df[feat].dropna().unique().tolist()
        categorical_options[feat] = sorted(unique_values, key=lambda x: str(x).lower())

def probability_to_color(prob):
    """
    Maps probability [0, 1] to color ranges:
    Green: 0-25% (low risk)
    Orange: 25-50% (medium risk)
    Red: 50-100% (high risk)
    """
    prob = max(0.0, min(1.0, prob))
    pct = prob * 100
    if pct < 25:
        return "rgb(144, 238, 144)"  # Light green
    elif pct < 50:
        return "rgb(255, 165, 0)"    # Orange
    else:
        return "rgb(255, 144, 144)"  # Light red

# ---- Input form with new layout ----
st.subheader("Enter Feature Values and Click Predict")
with st.form("manual_input"):
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.2])
    user_row = {}
    input_cols = [col1, col2, col3, col4]

    # Loop over model feature_names (includes 'h3_index')
    for i, feat in enumerate(feature_names):
        with input_cols[i % 4]:
            if feat in numeric_features:
                constraints = numeric_constraints.get(feat, {})
                min_val = constraints.get('min', 0)
                max_val = constraints.get('max', 100)
                step_val = constraints.get('step', 1)
                default_val = constraints.get('default', min_val)
                user_val = st.number_input(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    step=step_val,
                    value=default_val,
                    help=f"Range: {min_val} - {max_val}"
                )
            elif feat in categorical_options:
                user_val = st.selectbox(feat, categorical_options[feat])
            else:
                user_val = st.text_input(feat, value="", placeholder="type text")
            user_row[feat] = coerce_value(feat, str(user_val))

    with col5:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        prediction_placeholder = st.empty()

# ---- Prediction logic ----
if submitted:
    X_input = pd.DataFrame([user_row], columns=feature_names)
    y_prob = float(model.predict_proba(X_input)[:, 1][0])
    y_pred = int(y_prob >= 0.25)

    bg_color = probability_to_color(y_prob)
    label = "INJURY" if y_pred == 1 else "SAFE"
    icon = "⚠️" if y_pred == 1 else "✅"

    card_html = f"""
    <div style="
        background-color:{bg_color};
        width:100%;
        height:400px;
        border-radius:10px;
        padding:15px;
        text-align:center;
        color:black;
        font-family:sans-serif;
        margin-top:10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size:35px; margin-bottom:10px;">{icon}</div>
        <div style="font-size:24px; font-weight:bold; margin-bottom:10px;">{label}</div>
        <div style="font-size:16px; line-height:1.3;">
            The model estimates a <br><b style="font-size:18px;">{y_prob*100:.0f}%</b><br> probability of injury
        </div>
    </div>
    """
    with col5:
        prediction_placeholder.markdown(card_html, unsafe_allow_html=True)

# Create two columns for maps
col_left, col_right = st.columns([1, 1])

# --- Left column: Kepler Map (optional) ---
if KEPLER_PATH:
    with col_left:
        with open(KEPLER_PATH, "r", encoding="utf-8") as f:
            kepler_html = f.read()
        st.subheader("Kepler Map")
        html(kepler_html, height=500)

# --- Right column: Folium Map (optional) ---
if FOLIUM_PATH:
    with col_right:
        with open(FOLIUM_PATH, "r", encoding="utf-8") as f:
            folium_html = f.read()
        st.subheader("Folium Map")
        html(folium_html, height=400)

# --- Feature importance section ---
st.subheader("Feature Importance")
try:
    importances = model.get_feature_importance()
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
except Exception:
    fi_pretty = model.get_feature_importance(prettified=True)
    col_feat = [c for c in fi_pretty.columns if "Feature" in c][0]
    col_imp  = [c for c in fi_pretty.columns if "Import" in c][0]
    fi_df = fi_pretty.rename(columns={col_feat: "Feature", col_imp: "Importance"})[
        ["Feature", "Importance"]
    ]

fi_df = fi_df.sort_values("Importance", ascending=False)
top_n = st.slider("Show top features", 5, min(25, len(fi_df)), 15)
st.bar_chart(
    fi_df.head(top_n).set_index("Feature")["Importance"],
    use_container_width=True
)
