import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="AutoJudge – Problem Difficulty Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AutoJudge")
st.write("Predict the difficulty of a programming problem using its textual description.")

# --------------------------------------------------
# Load models (JOBLIB ONLY)
# --------------------------------------------------
@st.cache_resource
def load_models():
    # Classification artifacts
    clf_vectorizer = joblib.load("models/vectorizer.pkl")
    svd            = joblib.load("models/svd.pkl")
    clf_model      = joblib.load("models/lr_model.pkl")

    # Regression artifacts
    reg_vectorizer = joblib.load("models/tfidf_reg.pkl")
    reg_model      = joblib.load("models/ridge_reg.pkl")

    return clf_vectorizer, svd, clf_model, reg_vectorizer, reg_model


clf_vectorizer, svd, clf_model, reg_vectorizer, reg_model = load_models()

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def score_to_class(score):
    if score < 3:
        return "easy"
    elif score < 6:
        return "medium"
    else:
        return "hard"

# --------------------------------------------------
# UI – Required fields only
# --------------------------------------------------
st.subheader("📄 Enter Problem Details")

description = st.text_area(
    "Problem Description",
    height=160,
    placeholder="Describe the problem statement..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Describe the output format..."
)

predict_btn = st.button("🔮 Predict Difficulty")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict_btn:
    combined_text = f"{description} {input_desc} {output_desc}".strip()

    if combined_text == "":
        st.warning("Please enter at least one field before prediction.")
    else:
        # -------- Classification --------
        X_clf = clf_vectorizer.transform([combined_text])
        X_clf_svd = svd.transform(X_clf)
        pred_class = clf_model.predict(X_clf_svd)[0]

        # -------- Regression --------
        X_reg = reg_vectorizer.transform([combined_text])
        pred_score = reg_model.predict(X_reg)[0]
        pred_score = float(np.clip(pred_score, 0, 10))

        # -------- Display Results --------
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        st.write(f"### 🧠 Predicted Difficulty Class: **{pred_class.upper()}**")
        st.write(f"### 🔢 Predicted Difficulty Score: **{pred_score:.2f} / 10**")

        # -------- Consistency Check --------
        score_based_class = score_to_class(pred_score)

        if score_based_class != pred_class:
            st.warning(
                f"⚠️ The predicted score suggests **{score_based_class.upper()}** difficulty. "
                "This indicates the problem may be borderline between difficulty levels."
            )
        else:
            st.success("✅ Classification and score prediction are consistent.")

        st.markdown("---")
        st.caption(
            "Note: Predictions are based on the combined analysis of the problem description, "
            "input format, and output format."
        )
