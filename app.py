import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer Diagnostics",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for a professional "Card" look
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background-color: #f9f9f9;
            border: 1px solid #e6e6e6;
            padding: 10px;
            border-radius: 5px;
        }
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 2. Setup Paths
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "saved_models"

# 3. Header Section
c_head1, c_head2 = st.columns([1, 6])

with c_head2:
    st.title("Diagnostic AI Dashboard")
    st.markdown("Breast Cancer Prediction & Model Evaluation Suite")

st.markdown("---")

# 4. THE CONTROL DECK (Replaces Sidebar)
# We use an expander that is open by default.
with st.expander("⚙️ **Configuration & Data Upload**", expanded=True):
    col_conf1, col_conf2 = st.columns(2, gap="medium")
    
    with col_conf1:
        st.subheader("1. Select Model")
        if not MODEL_DIR.exists():
            st.error("⚠️ Directory 'saved_models/' not found.")
            st.stop()
            
        model_files = sorted([p.name for p in MODEL_DIR.glob("*.joblib")])
        if not model_files:
            st.error("⚠️ No models found.")
            st.stop()
            
        selected_model = st.selectbox("Choose a pre-trained classifier:", model_files)
        model_path = MODEL_DIR / selected_model
        
        try:
            model = joblib.load(model_path)
            st.caption(f"✅ Active Model: **{selected_model}**")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    with col_conf2:
        st.subheader("2. Upload Patient Data")
        uploaded_file = st.file_uploader("Upload CSV (Wisconsin Dataset format)", type=["csv"])

# --- Main Logic ---
if uploaded_file is None:
    # Show a placeholder message when empty
    st.info("👆 Please configure the model and upload a CSV file in the panel above to generate results.")
    st.stop()

# Load Data
df_input = pd.read_csv(uploaded_file)

# Feature Validation
from sklearn.datasets import load_breast_cancer
ref_data = load_breast_cancer()
feature_names = list(ref_data.feature_names)
missing_cols = [c for c in feature_names if c not in df_input.columns]

if missing_cols:
    st.error(f"❌ Input CSV is missing required columns:\n\n{', '.join(missing_cols[:5])}...")
    st.stop()

# Predict
X = df_input[feature_names].values
predictions = model.predict(X)
df_output = df_input.copy()
df_output["prediction"] = predictions

has_proba = hasattr(model, "predict_proba")
if has_proba:
    proba = model.predict_proba(X)[:, 1]
    df_output["probability_malignant"] = proba

# --- Results Presentation ---
st.markdown("### 📊 Analysis Results")
tab1, tab2, tab3 = st.tabs(["Patient Predictions", "Model Performance", "Raw Data"])

with tab1:
    # Summary Cards
    n_total = len(df_output)
    n_mal = sum(predictions)
    n_ben = n_total - n_mal
    
    # Use 4 columns for a wider spread of metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients", n_total)
    m2.metric("Benign Cases", n_ben)
    m3.metric("Malignant Cases", n_mal, delta_color="inverse")
    m4.metric("Risk Ratio", f"{(n_mal/n_total):.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled Dataframe
    cols_to_show = ["prediction"] 
    if has_proba:
        cols_to_show.append("probability_malignant")
    cols_to_show += feature_names[:5]
    
    display_df = df_output[cols_to_show]
    
    def highlight_malignant(s):
        return ['background-color: #ffe6e6' if v == 1 else '' for v in s]
    
    st.dataframe(
        display_df.style.apply(highlight_malignant, subset=['prediction']),
        use_container_width=True
    )
    
    csv = df_output.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results CSV", csv, "predictions.csv", "text/csv")

with tab2:
    if "target" in df_input.columns:
        y_true = df_input["target"].values
        y_pred = predictions

        # Metrics Row

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
        c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
        c4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2%}")
        c5.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.3f}")


        st.divider()

        # Layout: Confusion Matrix (Left) | AUC Gauge (Right)
        col_cm, col_auc = st.columns([1, 2])

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(3, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            st.pyplot(fig, use_container_width=False)

        with col_auc:
            st.markdown("**ROC / AUC Analysis**")
            if has_proba:
                auc = roc_auc_score(y_true, df_output["probability_malignant"])
                st.metric("AUC Score", f"{auc:.4f}")
                st.progress(auc)

                if auc > 0.9:
                    st.success("Excellent discrimination capability.")
                elif auc > 0.7:
                    st.warning("Acceptable discrimination capability.")
                else:
                    st.error("Poor model performance.")
            else:
                st.info("Model does not support probabilities.")

        st.divider()

        # ---- Classification Report (below the two-column layout) ----
        st.markdown("**Classification Report**")

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(3)

        st.dataframe(report_df, use_container_width=True)

    else:
        st.warning("⚠️ No 'target' column found in uploaded CSV. Metrics unavailable.")


with tab3:
    st.dataframe(df_input, use_container_width=True)