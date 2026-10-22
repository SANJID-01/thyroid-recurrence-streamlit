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
  .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
  [data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 34%, #ffffff 100%);}
  [data-testid="stHeader"] {background: transparent;}
  .hero {padding: 1.6rem 1.8rem; border-radius: 24px; color: #ffffff;
          background: linear-gradient(120deg, #075985 0%, #0f766e 52%, #2563eb 100%);
          box-shadow: 0 12px 30px rgba(15, 118, 110, .16); margin-bottom: 1rem;}
  .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.035em; color: #ffffff;}
  .hero p {margin: .55rem 0 0 0; font-size: 1rem; color: #e0f2fe;}
  .pill {display:inline-block; padding:.25rem .65rem; border:1px solid rgba(255,255,255,.35);
         border-radius:999px; font-size:.78rem; letter-spacing:.02em; margin-bottom:.6rem;}
  .section-title {font-size:1.14rem; font-weight:750; color:#0f172a; margin:.25rem 0 .3rem 0;}
  .muted {color:#64748b; font-size:.9rem;}
  .mini-card {padding: .95rem 1rem; border:1px solid #dbeafe; background:#ffffff;
               border-radius:16px; margin: .15rem 0 .6rem 0;}
  .warning-card {padding:.85rem 1rem; background:#fff7ed; border:1px solid #fed7aa;
                  border-radius:14px; color:#9a3412;}
  div.stButton > button[kind="primaryFormSubmit"] {border-radius:12px; font-weight:700; min-height:3rem;}
  div[data-testid="stForm"] {border:1px solid #dbeafe; border-radius:18px; padding:1.1rem; background:#ffffff;}
  div[data-testid="stMetric"] {background:#ffffff; padding:1rem; border:1px solid #dbeafe; border-radius:16px;}
  footer {visibility: hidden;}
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

st.caption("Primary research track excludes post-treatment `Response` and derived `Stage`. Patient identifiers and photos are not prediction features and are not saved by this app.")

st.markdown('<div class="section-title">01 · Privacy-conscious research input</div>', unsafe_allow_html=True)
st.markdown('This prototype does not collect names, patient identifiers, national ID numbers, or images. Use synthetic or appropriately de-identified data only.')
st.divider()
st.markdown('<div class="section-title">02 · Research model inputs</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Enter values for the available clinical variables. The patient ID, NID and photo are excluded from the model input.</div>', unsafe_allow_html=True)

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
                st.metric("Model-estimated recurrence score", f"{score:.1%}")
                st.progress(score, text="Model score visualisation only — not a clinical risk category")
            with out_right:
                st.markdown('<div class="mini-card"><b>Interpretation boundary</b><br><span class="muted">This number reflects the model output for the entered values. It is not a diagnosis, validated individual prognosis, or treatment recommendation.</span></div>', unsafe_allow_html=True)

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
                st.info(f"Local SHAP explanation is unavailable in this run ({type(exc).__name__}). The model score above is independent of this optional explanation.")

            st.markdown('<div class="warning-card"><b>Next step:</b> A qualified clinician must review the patient independently. This app does not give diagnosis, treatment, or follow-up advice.</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.info("Check that the model artifact, metadata, feature names, categorical values, and package versions all match the notebook that generated them.")

st.divider()
with st.expander("About this prototype and data handling"):
    st.write("Dataset:", meta.get("dataset", "Not specified"))
    st.write("Target column:", meta.get("target", "Not specified"))
    st.write("Model feature columns:", ", ".join(meta.get("feature_columns", [])))
    st.write("This prototype is intended for synthetic or appropriately de-identified research inputs only. A hosted app processes entered values in memory; do not enter identifiable patient information without institutional approval, access controls, secure hosting, and an approved data-governance plan.")
    st.write("This is a research software prototype. It has not been externally or prospectively validated and is not approved for clinical use.")

st.caption("ThyroidCare Research Prototype · For research demonstration only")
