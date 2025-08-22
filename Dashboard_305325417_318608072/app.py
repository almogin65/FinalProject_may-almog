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

st.set_page_config(page_title="Injury Score Only", layout="wide")

# === Top header: title (left) + score (right) ===
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.title("🚑 Injury Prediction")

with hdr_right:
    score_top = st.empty()      # upper-right metric
    caption_top = st.empty()    # small caption under the metric


# --- Upload trained model ---
with open("catboost_model.pkl", "rb") as model_file:
    # Load the model using pickle
    # Note: Using BytesIO is not necessary here since we are reading directly from a file
    # However, if you want to use BytesIO, you can uncomment the next two lines
    # import io
    # model_bytes = model_file.read()
    # model = pickle.load(io.BytesIO(model_bytes))
    
    model = pickle.load(model_file)

#upload csv
df=pd.read_csv('processed_data.csv',index_col=0).drop(columns='Injury Severity')
len(df.columns)
len(model.feature_names_)
# ---- Feature schema (your list) ----
feature_names = [
 'Agency Name','ACRS Report Type','Route Type','Collision Type','Weather',
 'Surface Condition','Light','Traffic Control','Driver Substance Abuse',
 'Driver At Fault','Driver Distracted By','Drivers License State',
 'Vehicle Damage Extent','Vehicle First Impact Location','Vehicle Body Type',
 'Vehicle Movement','Vehicle Going Dir','Speed Limit','Vehicle Year',
 'Vehicle Make','Crash Hour','Crash Day','Crash Month','h3_index'
]

# Input fields for the form (includes lat/lng but not h3_index since it's calculated)
input_fields = [
 'Agency Name','ACRS Report Type','Route Type','Collision Type','Weather',
 'Surface Condition','Light','Traffic Control','Driver Substance Abuse',
 'Driver At Fault','Driver Distracted By','Drivers License State',
 'Vehicle Damage Extent','Vehicle First Impact Location','Vehicle Body Type',
 'Vehicle Movement','Vehicle Going Dir','Speed Limit','Vehicle Year',
 'Vehicle Make','Crash Hour','Crash Day','Crash Month','Latitude','Longitude'
]



# Features to treat as numeric; others are categorical/text
numeric_features = {'Speed Limit','Vehicle Year','Crash Hour','Crash Day','Crash Month','Latitude','Longitude'}
# If your h3_index is numeric, add: numeric_features.add('h3_index')

# Define constraints for numeric features
numeric_constraints = {
    'Speed Limit': {'min': 0, 'max': 200, 'step': 5, 'default': 25},
    'Vehicle Year': {'min': 1980, 'max': 2025, 'step': 1, 'default': 2020},
    'Crash Hour': {'min': 0, 'max': 23, 'step': 1, 'default': 12},
    'Crash Day': {'min': 1, 'max': 31, 'step': 1, 'default': 15},
    'Crash Month': {'min': 1, 'max': 12, 'step': 1, 'default': 6},
    'Latitude': {'min': -90.0, 'max': 90.0, 'step': 0.000001, 'default': 39.2904},  # Default to Baltimore area
    'Longitude': {'min': -180.0, 'max': 180.0, 'step': 0.000001, 'default': -76.6122}  # Default to Baltimore area
}

# ---- Helpers ----
def coerce_value(name: str, val: str):
    """Convert input string to the right type or None."""
    val = val.strip()
    if val == "":
        return None  # let CatBoost handle missing
    if name in numeric_features:
        try:
            # Prefer int when appropriate; else float
            return int(val) if val.isdigit() or (val.startswith('-') and val[1:].isdigit()) else float(val)
        except ValueError:
            return None
    return val  # categorical/text

#add categorical option using unique values from the dataset
categorical_options = {}
for feat in feature_names:
    if feat not in numeric_features:
        # Get unique values from the dataset for categorical features and sort alphabetically
        unique_values = df[feat].dropna().unique().tolist()
        categorical_options[feat] = sorted(unique_values, key=lambda x: str(x).lower())

def probability_to_color(prob):
    """
    Maps probability [0, 1] to color ranges:
    Green: 0-25% (low risk)
    Orange: 25-50% (medium risk)  
    Red: 50-100% (high risk)
    """
    # Clamp prob to [0,1]
    prob = max(0.0, min(1.0, prob))
    
    # Convert to percentage for easier logic
    prob_percent = prob * 100
    
    if prob_percent < 25:
        # Green for low risk (0-25%)
        return "rgb(144, 238, 144)"  # Light green
    elif prob_percent < 50:
        # Orange for medium risk (25-50%)
        return "rgb(255, 165, 0)"    # Orange
    else:
        # Red for high risk (50-100%)
        return "rgb(255, 144, 144)"  # Light red

# ---- Input form with new layout ----
st.subheader("Enter Feature Values and Click Predict")
with st.form("manual_input"):
    # Create 5 columns: 4 for inputs + 1 for prediction result
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.2])
    
    user_row = {}
    
    # Distribute features across 4 columns
    input_cols = [col1, col2, col3, col4]
    
    for i, feat in enumerate(feature_names):
        with input_cols[i % 4]:  # Cycle through 4 columns instead of 5
            if feat in numeric_features:
                # Numeric input with constraints
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
                # Dropdown with predefined values
                user_val = st.selectbox(feat, categorical_options[feat])
            else:
                # Default text input for categorical with no predefined list
                user_val = st.text_input(feat, value="", placeholder="type text")
            
            user_row[feat] = coerce_value(feat, str(user_val))

    # Submit button in the prediction column
    with col5:
        st.write("")  # Add some spacing
        st.write("")  # Add more spacing to align with other inputs
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        # Placeholder for prediction result
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
    
    # Display prediction in the 5th column
    with col5:
        prediction_placeholder.markdown(card_html, unsafe_allow_html=True)

# Create two columns for maps
col_left, col_right = st.columns([1, 1])

# --- Left column: Kepler Map ---
with col_left:
    with open("kepler_map.html", "r", encoding="utf-8") as f:
        kepler_html = f.read()
    st.subheader("Kepler Map")
    html(kepler_html, height=500)

# --- Right column: Folium Map ---
with col_right:
    with open("my_folium_map.html", "r", encoding="utf-8") as f:
        folium_html = f.read()
    st.subheader("Folium Map")
    html(folium_html, height=400)

# --- Feature importance section ---
st.subheader("Feature Importance")

# Use CatBoost importances; align with your feature_names
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

# Optional: let the user choose Top N
top_n = st.slider("Show top features", 5, min(25, len(fi_df)), 15)
st.bar_chart(
    fi_df.head(top_n).set_index("Feature")["Importance"],
    use_container_width=True
)