import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
import sys
if str(BASE) not in sys.path: sys.path.insert(0, str(BASE))
from custom_transformers import DataFrameAgeScaler  # required to unpickle the trained pipeline
MODEL_PATH = BASE / "response_free_clinical_rf_pipeline.joblib"
META_PATH = BASE / "streamlit_metadata.json"

st.set_page_config(
    page_title="ThyroidCare Research | Recurrence Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Visual design --------------------
st.markdown("""
<style>
  :root {
    color-scheme: light;
    --tc-ink: #172033;
    --tc-muted: #475569;
    --tc-border: #cbd5e1;
    --tc-panel: #ffffff;
    --tc-control: #f8fafc;
  }
  html, body, [class*="css"], .stApp { color: var(--tc-ink) !important; }
  .stApp { background: linear-gradient(180deg, #eff6ff 0%, #f8fafc 35%, #ffffff 100%); }
  [data-testid="stAppViewContainer"] { background: transparent; }
  [data-testid="stHeader"] { background: rgba(255,255,255,.82); }
  .block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }
  .stApp p, .stApp li, .stApp label, .stApp small,
  .stApp [data-testid="stMarkdownContainer"],
  .stApp [data-testid="stWidgetLabel"],
  .stApp [data-testid="stWidgetLabel"] p,
  .stApp [data-testid="stCaptionContainer"],
  .stApp [data-testid="stCaptionContainer"] p,
  .stApp [data-testid="stExpander"] summary {
    color: var(--tc-ink) !important;
    opacity: 1 !important;
  }
  .hero { padding: 1.55rem 1.8rem; border-radius: 22px; color: #ffffff !important;
          background: linear-gradient(120deg, #075985 0%, #0f766e 52%, #2563eb 100%);
          box-shadow: 0 12px 30px rgba(15, 118, 110, .16); margin-bottom: 1rem; }
  .hero, .hero * { color: #ffffff !important; }
  .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.035em; line-height: 1.2; }
  .hero p { margin: .65rem 0 0 0; font-size: 1rem; color: #eff6ff !important; line-height: 1.55; }
  .pill { display:inline-block; padding:.32rem .7rem; border:1px solid rgba(255,255,255,.55);
          border-radius:999px; font-size:.78rem; font-weight:700; letter-spacing:.02em; margin-bottom:.7rem; }
  .section-title { font-size:1.15rem; font-weight:800; color:#0f172a !important; margin:.55rem 0 .35rem 0; }
  .muted { color:#475569 !important; font-size:.94rem; line-height:1.55; }
  .info-card { padding:.85rem 1rem; border:1px solid #bfdbfe; background:#eff6ff;
               border-left:5px solid #2563eb; border-radius:12px; color:#1e293b !important;
               line-height:1.55; margin:.3rem 0 .85rem 0; }
  .info-card * { color:#1e293b !important; }
  .mini-card { padding: 1rem 1.05rem; border:1px solid #cbd5e1; background:#ffffff;
               border-radius:14px; margin: .15rem 0 .6rem 0; color:#172033 !important; line-height:1.55; }
  .mini-card * { color:#172033 !important; }
  .warning-card { padding:.9rem 1rem; background:#fff7ed; border:1px solid #fdba74;
                  border-left:5px solid #ea580c; border-radius:13px; color:#7c2d12 !important;
                  line-height:1.6; margin:.7rem 0 1rem 0; }
  .warning-card * { color:#7c2d12 !important; }
  div[data-testid="stForm"] { border:1px solid #cbd5e1; border-radius:16px; padding:1.05rem 1.1rem;
                               background:#ffffff; box-shadow:0 4px 16px rgba(15,23,42,.045); }
  div[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] p,
  div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
    color:#172033 !important; font-weight:700 !important; font-size:.94rem !important;
  }
  div[data-baseweb="select"] > div { background:#f8fafc !important; border:1px solid #94a3b8 !important;
                                      min-height:2.75rem; }
  div[data-baseweb="select"] *, div[data-baseweb="select"] span,
  div[data-baseweb="select"] input { color:#111827 !important; opacity:1 !important; }
  div[data-baseweb="select"] svg { fill:#334155 !important; }
  div[data-testid="stNumberInput"] input { background:#f8fafc !important; color:#111827 !important;
                                             border:1px solid #94a3b8 !important; min-height:2.75rem; }
  div[data-testid="stNumberInput"] button { color:#1e293b !important; background:#e2e8f0 !important; }
  div[data-testid="stMetric"] { background:#ffffff; padding:1rem; border:1px solid #cbd5e1;
                                  border-radius:14px; }
  div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#172033 !important; }
  div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    border-radius:11px !important; font-weight:750 !important; min-height:3rem;
    font-size:1rem !important; border:1px solid #b91c1c !important;
    background:#dc2626 !important; color:#ffffff !important;
  }
  div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover,
  div.stButton > button:hover *, div[data-testid="stFormSubmitButton"] button:hover * {
    background:#b91c1c !important; border-color:#991b1b !important; color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
  }
  /* Streamlit versions may style the button's inner paragraph separately. */
  div.stButton > button, div.stButton > button *,
  div[data-testid="stFormSubmitButton"] button,
  div[data-testid="stFormSubmitButton"] button * {
    color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; opacity:1 !important;
  }
  /* Enforce the light form theme even when the browser or Cloud theme is dark. */
  [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:#f8fafc !important; color:#111827 !important; border-color:#94a3b8 !important;
  }
  [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color:#111827 !important; -webkit-text-fill-color:#111827 !important; opacity:1 !important;
  }
  [data-testid="stExpander"] details, [data-testid="stExpander"] details > summary,
  [data-testid="stExpander"] details > summary *, [data-testid="stExpander"] summary p {
    background:#ffffff !important; color:#172033 !important; -webkit-text-fill-color:#172033 !important;
    opacity:1 !important;
  }
  [data-testid="stExpander"] details { border:1px solid #cbd5e1 !important; border-radius:12px !important; }
  [data-testid="stProgressBar"] { margin-top:.35rem; }
  [data-testid="stDataFrame"] { border:1px solid #cbd5e1; border-radius:10px; overflow:hidden; }
  [data-testid="stAlert"] { border-radius:12px; }
  footer { visibility: hidden; }
  @media (max-width: 700px) {
    .block-container { padding: 1rem .75rem 2rem .75rem; }
    .hero { padding:1.2rem 1.1rem; border-radius:16px; }
    .hero h1 { font-size:1.75rem; }
    div[data-testid="stForm"] { padding:.8rem; }
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="pill">RESEARCH PROTOTYPE · NOT FOR CLINICAL USE</div>
  <h1>🩺 ThyroidCare</h1>
  <p>Thyroid cancer recurrence research model · structured inputs · transparent limitations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-card"><b>Research-only notice:</b> This unvalidated model is not a medical device. It must not be used to diagnose, choose treatment, set follow-up intervals, or make care decisions. A clinician must independently assess every patient. Do not enter real NID or identifiable patient data into this public demo.</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading research model…")
def load_assets():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file is missing: {MODEL_PATH.name}")
    if not META_PATH.is_file():
        raise FileNotFoundError(f"Metadata file is missing: {META_PATH.name}")
    with META_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    pipeline = joblib.load(MODEL_PATH)
    return pipeline, metadata

try:
    model, meta = load_assets()
except Exception as exc:
    st.error(f"The app could not load the model or metadata: {type(exc).__name__}: {exc}")
    st.stop()

st.markdown(
    '<div class="info-card"><b>Model scope:</b> This research track excludes post-treatment response and derived stage variables. '
    'Patient identifiers and photos are not prediction features.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">01 · Privacy-conscious research input</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-card">This prototype does not ask for names, patient identifiers, national ID numbers, or images. '
    '<b>Use synthetic or appropriately de-identified data only.</b></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-title">02 · Research model inputs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Enter the values requested below. Patient ID, NID, and photo are not used as model inputs.</div>',
    unsafe_allow_html=True,
)

with st.form("patient_inputs", clear_on_submit=False):
    row = {}
    age_min = int(meta.get("age_min", 0))
    age_max = int(meta.get("age_max", 120))
    if age_min > age_max:
        age_min, age_max = age_max, age_min
    default_age = max(age_min, min(age_max, int((age_min + age_max) / 2)))
    row["Age"] = st.number_input("Age (years)", min_value=age_min, max_value=age_max, value=default_age, step=1)

    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        st.error("No model feature columns were found in streamlit_metadata.json.")
        st.stop()

    other_columns = [c for c in feature_columns if c != "Age"]
    feature_cols = st.columns(2, gap="large")
    missing_options = []
    for i, col in enumerate(other_columns):
        options = meta.get("categorical_options", {}).get(col, [])
        with feature_cols[i % 2]:
            if not options:
                missing_options.append(col)
                st.error(f"No saved training options for {col}.")
            else:
                row[col] = st.selectbox(col, options=[str(x) for x in options], key=f"feature_{col}")
    submitted = st.form_submit_button("Run research prediction", type="primary", use_container_width=True)

if submitted:
    if missing_options:
        st.error("Prediction stopped because category options are missing for: " + ", ".join(missing_options))
    else:
        try:
            # Only model features are passed to the pipeline; identity/photo fields never enter this DataFrame.
            X_input = pd.DataFrame([{c: row[c] for c in feature_columns}], columns=feature_columns)
            if "Age" in X_input.columns:
                X_input["Age"] = pd.to_numeric(X_input["Age"], errors="raise").astype(int)
            pred_probs = model.predict_proba(X_input)
            classes = list(getattr(model, "classes_", [0, 1]))
            positive_matches = [i for i, c in enumerate(classes) if str(c) == "1"]
            positive_index = positive_matches[0] if positive_matches else min(1, pred_probs.shape[1] - 1)
            score = float(pred_probs[0, positive_index])
            score = min(1.0, max(0.0, score))

            st.divider()
            st.markdown('<div class="section-title">03 · Research model output</div>', unsafe_allow_html=True)
            out_left, out_right = st.columns([1, 1], gap="large")
            with out_left:
                st.metric("Unvalidated model recurrence score", f"{score:.1%}")
                st.progress(score, text="Research model output only — not a validated clinical probability")
            with out_right:
                st.markdown('<div class="mini-card"><b>Interpretation boundary</b><br><span class="muted">This is the model score for the entered values, not a validated real-world recurrence probability. It is not a diagnosis, individual prognosis, or treatment recommendation.</span></div>', unsafe_allow_html=True)

            # Optional local SHAP. Failure is reported as unavailable, never as zero contribution.
            try:
                import shap
                steps = getattr(model, "named_steps", {})
                prep = steps.get("prep") if hasattr(steps, "get") else None
                estimator = steps.get("model") if hasattr(steps, "get") else None
                if prep is None or estimator is None:
                    raise RuntimeError("Expected pipeline steps 'prep' and 'model' were not found.")
                transformed = prep.transform(X_input)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_values = np.asarray(shap_values)
                if shap_values.ndim == 3:
                    class_index = min(1, shap_values.shape[-1] - 1)
                    shap_row = shap_values[0, :, class_index]
                elif shap_values.ndim == 2:
                    shap_row = shap_values[0]
                else:
                    raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")
                try:
                    feature_names = list(prep.get_feature_names_out())
                except Exception:
                    feature_names = [f"Encoded feature {i+1}" for i in range(len(shap_row))]
                if len(feature_names) != len(shap_row):
                    feature_names = [f"Encoded feature {i+1}" for i in range(len(shap_row))]
                contributions = pd.DataFrame({"Feature": feature_names, "SHAP contribution": shap_row})
                contributions["absolute_contribution"] = contributions["SHAP contribution"].abs()
                contributions = contributions.sort_values("absolute_contribution", ascending=False).head(8).drop(columns="absolute_contribution")
                st.subheader("Local explanation · SHAP")
                st.caption("Explains how this model behaved for this input; it does not explain causality or prove clinical risk factors.")
                st.dataframe(contributions, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.info(f"Local SHAP explanation could not be generated ({type(exc).__name__}). The score above is independent of this optional explanation. Check the deployment logs if this persists.")

            st.markdown('<div class="warning-card"><b>Next step:</b> A qualified clinician must review the patient independently. This app does not give diagnosis, treatment, or follow-up advice.</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.info("Check that the model artifact, metadata, feature names, categorical values, and package versions all match the notebook that generated them.")

st.divider()
st.markdown('<div class="section-title">About this prototype and data handling</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.write("**Dataset:**", meta.get("dataset", "Not specified"))
    st.write("**Target column:**", meta.get("target", "Not specified"))
    st.write("**Model feature columns:**", ", ".join(meta.get("feature_columns", [])))
    st.write("This prototype is intended for synthetic or appropriately de-identified research inputs only. A hosted app processes entered values in memory; do not enter identifiable patient information without institutional approval, access controls, secure hosting, and an approved data-governance plan.")
    st.write("This is a research software prototype. It has not been externally or prospectively validated and is not approved for clinical use.")

st.caption("ThyroidCare Research Prototype · For research demonstration only")
