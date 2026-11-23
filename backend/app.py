"""
PredictiVision-AI — Streamlit Dashboard
A visual interface for the Predictive Maintenance ML project.
Run with:  streamlit run backend/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Make backend/src importable ────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiVision-AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* gradient header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 2.4rem; font-weight: 700; }
.main-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.8; }

/* metric cards */
.metric-card {
    background: #1e1e2e;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    color: white;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
.metric-card .label { font-size: 0.85rem; color: #aaa; margin-top: 0.3rem; }

/* prediction result */
.result-failure {
    background: linear-gradient(135deg, #c62828, #b71c1c);
    color: white; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-normal {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: white; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-failure h2, .result-normal h2 { margin: 0; font-size: 1.8rem; }
.result-failure p,  .result-normal p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.9; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers / Cached Loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_model_comparison():
    csv_path = os.path.join(FIGURES_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return df
    return None


@st.cache_resource
def load_traditional_model(model_key: str):
    path = os.path.join(MODELS_DIR, f"{model_key}_best.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_scaler():
    p = os.path.join(MODELS_DIR, "scaler.joblib")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_resource
def load_dl_scaler():
    p = os.path.join(MODELS_DIR, "dl_scaler.joblib")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_data
def load_dataset():
    csv_path = os.path.join(DATA_DIR, "ai4i2020.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        rename = {
            "UDI": "udi", "Product ID": "product_id", "Type": "type",
            "Air temperature [K]": "air_temperature",
            "Process temperature [K]": "process_temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
            "Machine failure": "machine_failure",
            "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf"
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df
    return None


def read_meta():
    path = os.path.join(MODELS_DIR, "model_meta.txt")
    meta = {}
    if os.path.exists(path):
        for line in open(path).readlines():
            k, v = line.strip().split(": ", 1)
            meta[k] = v
    return meta


# DL model definitions (must match train.py exactly)
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, seq_len=10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64 * (seq_len // 2), 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)


@st.cache_resource
def load_dl_model(model_key: str, input_dim: int = 6, seq_len: int = 10):
    path = os.path.join(MODELS_DIR, f"{model_key}_best.pt")
    if not os.path.exists(path):
        return None
    if model_key == "lstm":
        m = LSTMClassifier(input_dim)
    elif model_key == "gru":
        m = GRUClassifier(input_dim)
    else:  # cnn1d
        m = CNN1DClassifier(input_dim, seq_len)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


TRAD_MODELS = {
    "XGBoost":             "xgboost",
    "LightGBM":            "lightgbm",
    "Random Forest":       "random_forest",
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "SVM":                 "svm",
}
DL_MODELS = {
    "LSTM":   "lstm",
    "GRU":    "gru",
    "1D CNN": "cnn1d",
}
ALL_MODELS = {**TRAD_MODELS, **DL_MODELS}

METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1_score":  "F1-Score",
    "roc_auc":   "ROC-AUC",
    "pr_auc":    "PR-AUC",
}

NUM_FEATURES = [
    "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear",
    "temp_difference", "power_index", "overstrain_index",
    "air_temperature_roll_mean_5", "air_temperature_roll_std_5",
    "process_temperature_roll_mean_5", "process_temperature_roll_std_5",
    "rotational_speed_roll_mean_5", "rotational_speed_roll_std_5",
    "torque_roll_mean_5", "torque_roll_std_5",
    "tool_wear_roll_mean_5", "tool_wear_roll_std_5",
    "rotational_speed_lag_1", "rotational_speed_lag_2",
    "torque_lag_1", "torque_lag_2",
    "air_temperature_lag_1", "air_temperature_lag_2",
    "temp_difference_lag_1", "temp_difference_lag_2",
]
FEATURE_COLS = ["type_encoded"] + NUM_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🏭 PredictiVision-AI")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Live Prediction", "📊 Model Comparison", "📈 Sensor Explorer", "🧠 Explainability"],
    label_visibility="collapsed",
)

meta = read_meta()
best_model_name = meta.get("best_model_name", "XGBoost")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best model:** `{best_model_name}`")
st.sidebar.markdown(f"**Task:** Binary failure classification")
st.sidebar.markdown(f"**Dataset:** AI4I 2020 (10 000 rows)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════

if PAGE == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🏭 PredictiVision-AI</h1>
        <p>Manufacturing Predictive Maintenance · AI4I 2020 Dataset · 9 Models Trained</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    comp = load_model_comparison()
    df_data = load_dataset()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="value">9</div><div class="label">Models Trained</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="value">10K</div><div class="label">Sensor Records</div></div>""",
            unsafe_allow_html=True)
    with c3:
        best_f1 = f"{comp['f1_score'].max():.4f}" if comp is not None else "0.8077"
        st.markdown(f"""<div class="metric-card">
            <div class="value">{best_f1}</div><div class="label">Best F1-Score</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="value">3.4%</div><div class="label">Failure Rate</div></div>""",
            unsafe_allow_html=True)
    with c5:
        st.markdown("""<div class="metric-card">
            <div class="value">27</div><div class="label">Engineered Features</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick model comparison bar ─────────────────────────────────────────────
    if comp is not None:
        st.subheader("📊 Model Performance at a Glance")
        df_sorted = comp.sort_values("f1_score", ascending=False).reset_index()
        fig = px.bar(
            df_sorted, x="model_name", y=["f1_score", "recall", "precision"],
            barmode="group",
            color_discrete_sequence=["#4fc3f7", "#81c784", "#ffb74d"],
            labels={"value": "Score", "model_name": "Model", "variable": "Metric"},
            title="F1 / Recall / Precision by Model",
        )
        fig.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", legend_title_text="Metric",
            xaxis_tickangle=-25, height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Failure modes breakdown ────────────────────────────────────────────────
    if df_data is not None:
        st.subheader("🔍 Failure Mode Distribution")
        col_l, col_r = st.columns(2)
        with col_l:
            failure_modes = {
                "Tool Wear (TWF)": df_data.get("twf", pd.Series([0])).sum(),
                "Heat Dissipation (HDF)": df_data.get("hdf", pd.Series([0])).sum(),
                "Power Failure (PWF)": df_data.get("pwf", pd.Series([0])).sum(),
                "Overstrain (OSF)": df_data.get("osf", pd.Series([0])).sum(),
                "Random (RNF)": df_data.get("rnf", pd.Series([0])).sum(),
            }
            fig_pie = px.pie(
                values=list(failure_modes.values()),
                names=list(failure_modes.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Failure Causes Breakdown",
                hole=0.4,
            )
            fig_pie.update_layout(
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="white", height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            type_counts = df_data["type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            type_full = {"L": "Low (50%)", "M": "Medium (30%)", "H": "High (20%)"}
            type_counts["Type Label"] = type_counts["Type"].map(type_full)
            fig_bar = px.bar(
                type_counts, x="Type Label", y="Count",
                color="Type Label",
                color_discrete_sequence=["#4fc3f7", "#81c784", "#ffb74d"],
                title="Tool Quality Type Distribution",
            )
            fig_bar.update_layout(
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="white", showlegend=False, height=320,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Pipeline diagram ───────────────────────────────────────────────────────
    st.subheader("⚙️ Project Pipeline")
    steps = [
        ("📥 Raw Data", "ai4i2020.csv\n10 000 rows"),
        ("🧹 Preprocess", "Clean · Scale\nChronological split"),
        ("🔧 Feature Eng.", "Physics · Rolling\nLag features"),
        ("🤖 Train Models", "6 Traditional ML\n3 Deep Learning"),
        ("📊 Evaluate", "F1 · Recall\nAUC · Confusion"),
        ("💡 Explain", "SHAP\nPermutation"),
        ("🔮 Predict", "Live inference\nany model"),
    ]
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        col.markdown(
            f"<div style='background:#1e1e2e;border:1px solid #2d2d44;border-radius:8px;"
            f"padding:0.8rem;text-align:center;color:white;height:110px'>"
            f"<b>{title}</b><br><small style='color:#aaa'>{desc.replace(chr(10),'<br>')}</small></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "🔮 Live Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Live Failure Prediction</h1>
        <p>Enter sensor readings to predict machine failure probability in real time</p>
    </div>
    """, unsafe_allow_html=True)

    scaler    = load_scaler()
    dl_scaler = load_dl_scaler()

    if scaler is None:
        st.error("⚠️ Scaler not found. Please run `python src/train.py` first to train the models.")
        st.stop()

    # ── Model selector ─────────────────────────────────────────────────────────
    sel_model_name = st.selectbox(
        "Select Model",
        list(ALL_MODELS.keys()),
        index=0,
        help="Choose which trained model to use for prediction",
    )
    model_key = ALL_MODELS[sel_model_name]
    is_dl     = model_key in ["lstm", "gru", "cnn1d"]

    st.markdown("---")
    st.subheader("🎚️ Sensor Input")
    st.caption("Adjust sliders to match the current machine sensor readings. Each row represents one time step.")

    # ── Single reading or batch ────────────────────────────────────────────────
    input_mode = st.radio("Input mode", ["Single reading (use sliders)", "Paste CSV / manual table"], horizontal=True)

    def build_single_reading():
        c1, c2, c3 = st.columns(3)
        with c1:
            tool_type         = st.selectbox("Tool Type", ["L", "M", "H"], help="L=Low quality, M=Medium, H=High")
            air_temp          = st.slider("Air Temperature (K)", 295.0, 305.0, 298.1, 0.1)
        with c2:
            proc_temp         = st.slider("Process Temperature (K)", 305.0, 315.0, 308.6, 0.1)
            rot_speed         = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
        with c3:
            torque_val        = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.5)
            tool_wear_val     = st.slider("Tool Wear (min)", 0, 250, 60)

        rows = []
        # Build 10-row history by slightly varying the last reading (simulate history)
        for i in range(10):
            noise = (10 - i) * 0.01
            rows.append({
                "type": tool_type,
                "air_temperature":      air_temp + noise,
                "process_temperature":  proc_temp + noise * 0.5,
                "rotational_speed":     rot_speed - i * 5,
                "torque":               torque_val + i * 0.3,
                "tool_wear":            max(0, tool_wear_val - (10 - i) * 2),
            })
        return pd.DataFrame(rows)

    if input_mode == "Single reading (use sliders)":
        history_df = build_single_reading()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Simulated sensor history (10 rows used for DL models)"):
            st.dataframe(history_df.style.format(precision=2), use_container_width=True)
    else:
        default_csv = (
            "type,air_temperature,process_temperature,rotational_speed,torque,tool_wear\n"
            + "\n".join(
                f"L,{298.1 + i*0.1:.1f},{308.2 + i*0.1:.1f},{1500 - i*20},{40.0 + i*1.5:.1f},{60 + i*2}"
                for i in range(10)
            )
        )
        raw_csv = st.text_area("Paste sensor history (CSV, 10+ rows recommended):", default_csv, height=220)
        try:
            from io import StringIO
            history_df = pd.read_csv(StringIO(raw_csv))
            st.success(f"✅ Parsed {len(history_df)} rows")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

    # ── Run prediction ─────────────────────────────────────────────────────────
    if st.button("⚡ Predict Failure", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            try:
                # ── feature engineering (mirrors predict.py) ───────────────
                df = history_df.copy()
                df["temp_difference"]  = df["process_temperature"] - df["air_temperature"]
                df["power_index"]      = df["rotational_speed"] * df["torque"]
                df["overstrain_index"] = df["tool_wear"] * df["torque"]

                win = 5
                for col in ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]:
                    df[f"{col}_roll_mean_{win}"] = df[col].rolling(win, min_periods=1).mean()
                    df[f"{col}_roll_std_{win}"]  = df[col].rolling(win, min_periods=1).std().fillna(0)

                for col in ["rotational_speed", "torque", "air_temperature", "temp_difference"]:
                    df[f"{col}_lag_1"] = df[col].shift(1).bfill()
                    df[f"{col}_lag_2"] = df[col].shift(2).bfill()

                type_map = {"L": 0, "M": 1, "H": 2}
                df["type_encoded"] = df["type"].map(type_map)

                df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])
                X_latest = df[FEATURE_COLS].iloc[[-1]]

                if not is_dl:
                    model = load_traditional_model(model_key)
                    if model is None:
                        st.error(f"Model file `{model_key}_best.joblib` not found. Run training first.")
                        st.stop()
                    pred = int(model.predict(X_latest)[0])
                    prob = float(model.predict_proba(X_latest)[0][1]) if hasattr(model, "predict_proba") else float(pred)
                else:
                    if dl_scaler is None:
                        st.error("DL scaler not found. Run training first.")
                        st.stop()
                    dl_core = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
                    for col in dl_core:
                        df[f"{col}_scaled"] = dl_scaler.transform(history_df[[col]])

                    dl_cols = ["type_encoded"] + [f"{c}_scaled" for c in dl_core]
                    seq_len = 10
                    if len(df) < seq_len:
                        pad = seq_len - len(df)
                        df = pd.concat([df.iloc[[0]] * pad, df], ignore_index=True)

                    X_seq = df[dl_cols].values[-seq_len:].reshape(1, seq_len, len(dl_cols))
                    X_t   = torch.tensor(X_seq, dtype=torch.float32)

                    m = load_dl_model(model_key, input_dim=len(dl_cols), seq_len=seq_len)
                    if m is None:
                        st.error(f"DL model `{model_key}_best.pt` not found. Run training first.")
                        st.stop()
                    with torch.no_grad():
                        logits = m(X_t)
                        prob   = torch.sigmoid(logits).item()
                    pred = 1 if prob >= 0.5 else 0

                # ── Results ───────────────────────────────────────────────
                col_res, col_gauge = st.columns([1, 1])
                with col_res:
                    if pred == 1:
                        st.markdown(f"""
                        <div class="result-failure">
                            <h2>⚠️ FAILURE PREDICTED</h2>
                            <p>Failure probability: <strong>{prob:.1%}</strong></p>
                            <p>Schedule maintenance immediately</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal">
                            <h2>✅ OPERATING NORMALLY</h2>
                            <p>Failure probability: <strong>{prob:.1%}</strong></p>
                            <p>No immediate action required</p>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prediction", "FAILURE" if pred else "NORMAL")
                    m2.metric("Failure Probability", f"{prob:.1%}")
                    m3.metric("Model Used", sel_model_name)

                with col_gauge:
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Failure Probability (%)", "font": {"size": 16, "color": "white"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "white"},
                            "bar":  {"color": "#e53935" if prob >= 0.5 else "#43a047"},
                            "steps": [
                                {"range": [0, 30],  "color": "#1b5e20"},
                                {"range": [30, 60], "color": "#f57f17"},
                                {"range": [60, 100],"color": "#b71c1c"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.75, "value": 50,
                            },
                        },
                        number={"suffix": "%", "font": {"color": "white"}},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="#1e1e2e", font_color="white",
                        height=300, margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "📊 Model Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Comparison</h1>
        <p>Compare all 9 trained models across key performance metrics</p>
    </div>
    """, unsafe_allow_html=True)

    comp = load_model_comparison()
    if comp is None:
        st.error("model_comparison.csv not found in figures/. Run `python src/train.py` to generate it.")
        st.stop()

    comp_pct = comp.copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        comp_pct[col] = (comp_pct[col] * 100).round(2)

    # ── Metric selector ────────────────────────────────────────────────────────
    metric_choice = st.selectbox(
        "Sort by metric",
        list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        index=3,  # F1
    )

    df_sorted = comp_pct.sort_values(metric_choice, ascending=False).reset_index()

    # ── Interactive bar chart ──────────────────────────────────────────────────
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4", "#ff8a65", "#a5d6a7", "#90caf9"]
    fig_comp = px.bar(
        df_sorted, x="model_name", y=metric_choice,
        color="model_name",
        color_discrete_sequence=colors,
        text=metric_choice,
        labels={"model_name": "Model", metric_choice: f"{METRIC_LABELS[metric_choice]} (%)"},
        title=f"Models ranked by {METRIC_LABELS[metric_choice]}",
    )
    fig_comp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_comp.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="white", showlegend=False,
        yaxis_range=[0, 105], height=430, xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Radar chart ────────────────────────────────────────────────────────────
    st.subheader("🕸️ Multi-metric Radar")
    radar_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    radar_labels  = [METRIC_LABELS[m] for m in radar_metrics]

    models_to_show = st.multiselect(
        "Models to compare",
        list(comp_pct.index),
        default=list(comp_pct.index[:4]),
    )

    fig_radar = go.Figure()
    palette   = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4", "#ff8a65", "#a5d6a7", "#90caf9"]
    for i, model in enumerate(models_to_show):
        vals = [comp_pct.loc[model, m] for m in radar_metrics] + [comp_pct.loc[model, radar_metrics[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            fill="toself", name=model,
            line_color=palette[i % len(palette)],
            opacity=0.75,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105], color="white", gridcolor="#2d2d44"),
            angularaxis=dict(color="white", gridcolor="#2d2d44"),
            bgcolor="#1e1e2e",
        ),
        paper_bgcolor="#1e1e2e", font_color="white",
        legend=dict(bgcolor="#1e1e2e"), height=480,
        title="Multi-metric Radar Comparison",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Full table ─────────────────────────────────────────────────────────────
    st.subheader("📋 Full Results Table")
    display_df = df_sorted.set_index("model_name").rename(columns=METRIC_LABELS)
    st.dataframe(
        display_df.style
            .format("{:.2f}%")
            .highlight_max(axis=0, color="#1b5e20")
            .highlight_min(axis=0, color="#4a1515"),
        use_container_width=True,
        height=380,
    )

    # ── Saved figure images ────────────────────────────────────────────────────
    st.subheader("🖼️ Saved Evaluation Plots")
    img_tabs = st.tabs([m for m in ALL_MODELS.keys()])
    for tab, model_name in zip(img_tabs, ALL_MODELS.keys()):
        with tab:
            key = model_name.lower().replace(" ", "_")
            cm_path  = os.path.join(FIGURES_DIR, f"{key}_confusion_matrix.png")
            cur_path = os.path.join(FIGURES_DIR, f"{key}_curves.png")
            c_l, c_r = st.columns(2)
            with c_l:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                else:
                    st.info("Confusion matrix not found")
            with c_r:
                if os.path.exists(cur_path):
                    st.image(cur_path, caption="ROC & PR Curves", use_container_width=True)
                else:
                    st.info("Curves plot not found")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Sensor Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "📈 Sensor Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Sensor Data Explorer</h1>
        <p>Visualise historical sensor trends, distributions, and correlations</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found at data/ai4i2020.csv")
        st.stop()

    SENSOR_COLS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]

    # ── Filters ────────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        type_filter = st.multiselect("Tool Type", ["L", "M", "H"], default=["L", "M", "H"])
    with col_f2:
        row_range = st.slider("Row range (time)", 0, len(df) - 1, (0, min(2000, len(df) - 1)))

    df_filt = df[df["type"].isin(type_filter)].iloc[row_range[0]:row_range[1]].copy()
    df_filt = df_filt.reset_index(drop=True)
    df_filt["index"] = df_filt.index

    if df_filt.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # ── Time-series chart ──────────────────────────────────────────────────────
    st.subheader("📉 Sensor Trends Over Time")
    sensor_sel = st.selectbox("Sensor to plot", SENSOR_COLS, index=0)

    fig_ts = go.Figure()
    # Colour-code by failure
    normal_mask  = df_filt["machine_failure"] == 0
    failure_mask = df_filt["machine_failure"] == 1

    fig_ts.add_trace(go.Scatter(
        x=df_filt.loc[normal_mask, "index"],
        y=df_filt.loc[normal_mask, sensor_sel],
        mode="lines", name="Normal",
        line=dict(color="#4fc3f7", width=1),
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_filt.loc[failure_mask, "index"],
        y=df_filt.loc[failure_mask, sensor_sel],
        mode="markers", name="Failure",
        marker=dict(color="#ef5350", size=8, symbol="x"),
    ))
    fig_ts.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="white", height=360,
        xaxis_title="Record index", yaxis_title=sensor_sel,
        title=f"{sensor_sel} — Normal vs Failure",
        legend=dict(bgcolor="#1e1e2e"),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Distribution & Correlation ─────────────────────────────────────────────
    col_dist, col_corr = st.columns(2)

    with col_dist:
        st.subheader("📊 Distribution by Failure")
        dist_sensor = st.selectbox("Sensor", SENSOR_COLS, key="dist_sel", index=2)
        fig_hist = px.histogram(
            df_filt, x=dist_sensor, color="machine_failure",
            nbins=50, barmode="overlay",
            color_discrete_map={0: "#4fc3f7", 1: "#ef5350"},
            labels={"machine_failure": "Failure"},
            title=f"{dist_sensor} distribution",
            opacity=0.7,
        )
        fig_hist.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_corr:
        st.subheader("🔗 Correlation Heatmap")
        corr_cols = SENSOR_COLS + ["machine_failure"]
        corr_m    = df_filt[corr_cols].corr()
        fig_corr  = px.imshow(
            corr_m, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Sensor Correlation Matrix",
            aspect="auto",
        )
        fig_corr.update_layout(
            paper_bgcolor="#1e1e2e", font_color="white", height=320,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Scatter ────────────────────────────────────────────────────────────────
    st.subheader("🔵 Scatter Plot")
    s1, s2, s3 = st.columns(3)
    with s1:
        x_col = st.selectbox("X axis", SENSOR_COLS, index=3)  # torque
    with s2:
        y_col = st.selectbox("Y axis", SENSOR_COLS, index=4)  # tool_wear
    with s3:
        sample_n = st.slider("Sample size", 100, min(5000, len(df_filt)), min(2000, len(df_filt)), 100)

    df_sample = df_filt.sample(n=sample_n, random_state=42) if len(df_filt) > sample_n else df_filt
    fig_sc = px.scatter(
        df_sample, x=x_col, y=y_col,
        color=df_sample["machine_failure"].map({0: "Normal", 1: "Failure"}),
        color_discrete_map={"Normal": "#4fc3f7", "Failure": "#ef5350"},
        opacity=0.6,
        title=f"{x_col} vs {y_col} (coloured by failure)",
        labels={"color": "Status"},
    )
    fig_sc.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="white", height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Summary stats ──────────────────────────────────────────────────────────
    st.subheader("📋 Summary Statistics")
    st.dataframe(df_filt[SENSOR_COLS + ["machine_failure"]].describe().T.style.format("{:.3f}"),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Explainability
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "🧠 Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Explainability (XAI)</h1>
        <p>Understand why the model makes each prediction using SHAP and feature importance</p>
    </div>
    """, unsafe_allow_html=True)

    xai_tabs = st.tabs(["📊 Feature Importance", "🌈 SHAP Summary", "📌 Permutation Importance"])

    with xai_tabs[0]:
        st.subheader("XGBoost — Tree-based Feature Importance")
        fi_path = os.path.join(FIGURES_DIR, "xgboost_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Feature importance plot not found. Run `python src/train.py` to generate it.")

        st.markdown("""
**Key insights:**
- `overstrain_index` (tool wear × torque) is the **top predictor** — directly captures Overstrain Failure (OSF).
- `power_index` (rotational speed × torque) captures Power Failure (PWF) thresholds.
- `temp_difference` (process − air temp) flags Heat Dissipation Failure (HDF) when it drops below 8.6 K.
        """)

    with xai_tabs[1]:
        st.subheader("SHAP Summary Plot")
        shap_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
        bar_path  = os.path.join(FIGURES_DIR, "shap_bar_plot.png")
        c1, c2    = st.columns(2)
        with c1:
            if os.path.exists(shap_path):
                st.image(shap_path, caption="SHAP Beeswarm (feature impact direction)", use_container_width=True)
            else:
                st.info("SHAP summary plot not found.")
        with c2:
            if os.path.exists(bar_path):
                st.image(bar_path, caption="SHAP Bar (mean absolute impact)", use_container_width=True)
            else:
                st.info("SHAP bar plot not found.")

        st.markdown("""
**How to read SHAP:**
- **Red dots** (high feature value) pushed **right** = that feature pushes prediction toward *failure*.
- **Blue dots** pushed **left** = low feature value reduces failure probability.
- High `overstrain_index` values strongly push the model toward predicting failure.
        """)

    with xai_tabs[2]:
        st.subheader("Permutation Feature Importance")
        perm_path = os.path.join(FIGURES_DIR, "xgboost_permutation_importance.png")
        if os.path.exists(perm_path):
            st.image(perm_path, use_container_width=True)
        else:
            st.info("Permutation importance plot not found.")

        st.markdown("""
**How permutation importance works:**
Randomly shuffle the values of one feature and measure how much the model's F1-score drops.
A large drop = the model genuinely relies on that feature.
- Shuffling `overstrain_index` and `torque` causes the **largest F1 drop**, confirming they are critical.
        """)

    # ── Why recall matters ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚖️ Why Recall > Precision in Predictive Maintenance")
    col_a, col_b = st.columns(2)
    with col_a:
        st.error("""
**❌ False Negative (missed failure)**
Model says "normal" but machine fails.
- Unexpected breakdown
- Damaged equipment
- Halted production line
- **Cost: $10K – $1M+**
        """)
    with col_b:
        st.success("""
**✅ False Positive (unnecessary inspection)**
Model says "failure" but machine is fine.
- Technician inspects a healthy machine
- Minor labor cost
- **Cost: ~$50 – $200**
        """)

    # ── Cost comparison chart ──────────────────────────────────────────────────
    comp = load_model_comparison()
    if comp is not None:
        fig_rc = px.scatter(
            comp.reset_index(),
            x="recall", y="precision",
            size="f1_score", text="model_name",
            color="f1_score",
            color_continuous_scale="Viridis",
            title="Recall vs Precision (bubble size = F1-Score)",
            labels={"recall": "Recall →  (catching failures)", "precision": "Precision →  (fewer false alarms)"},
        )
        fig_rc.update_traces(textposition="top center")
        fig_rc.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", height=430,
        )
        st.plotly_chart(fig_rc, use_container_width=True)
