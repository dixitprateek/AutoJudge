import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AutoJudge – Problem Difficulty Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AutoJudge")
st.write("Predict the difficulty of a programming problem using its description.")

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    # Classification artifacts
    with open("models/vectorizer.pkl", "rb") as f:
        clf_vectorizer = pickle.load(f)

    with open("models/svd.pkl", "rb") as f:
        svd = pickle.load(f)

    with open("models/lr_model.pkl", "rb") as f:
        clf_model = pickle.load(f)

    # Regression artifacts
    with open("models/tfidf_reg.pkl", "rb") as f:
        reg_vectorizer = pickle.load(f)

    with open("models/ridge_reg.pkl", "rb") as f:
        reg_model = pickle.load(f)

    return clf_vectorizer, svd, clf_model, reg_vectorizer, reg_model


clf_vectorizer, svd, clf_model, reg_vectorizer, reg_model = load_models()

# ----------------------------
# Helper functions
# ----------------------------
def score_to_class(score):
    if score < 3:
        return "easy"
    elif score < 6:
        return "medium"
    else:
        return "hard"


# ----------------------------
# UI
# ----------------------------
st.subheader("📄 Problem Description")

user_input = st.text_area(
    "Paste the problem title + description here:",
    height=250,
    placeholder="Example: Given a graph with N nodes and M edges, find the shortest path..."
)

predict_btn = st.button("🔮 Predict Difficulty")

# ----------------------------
# Prediction
# ----------------------------
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter a problem description.")
    else:
        # ---- Classification ----
        X_clf = clf_vectorizer.transform([user_input])
        X_clf_svd = svd.transform(X_clf)
        pred_class = clf_model.predict(X_clf_svd)[0]

        # ---- Regression ----
        X_reg = reg_vectorizer.transform([user_input])
        pred_score = reg_model.predict(X_reg)[0]
        pred_score = float(np.clip(pred_score, 0, 10))

        # ---- Display results ----
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        st.write(f"### 🧠 Predicted Difficulty Class: **{pred_class.upper()}**")
        st.write(f"### 🔢 Predicted Difficulty Score: **{pred_score:.2f} / 10**")

        # ---- Consistency check ----
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
            "Note: Predictions are based solely on textual analysis of the problem description."
        )
