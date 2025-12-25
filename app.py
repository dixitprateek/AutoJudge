import streamlit as st
import joblib

# Load models
vectorizer = joblib.load("vectorizer.pkl")
svd = joblib.load("svd.pkl")
clf = joblib.load("lr_model.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge: Problem Difficulty Classifier")
st.write("Paste the problem details to predict difficulty level.")

problem_desc = st.text_area("📘 Problem Description", height=200)
input_desc = st.text_area("📥 Input Description", height=150)
output_desc = st.text_area("📤 Output Description", height=150)

if st.button("🔮 Predict Difficulty"):
    if not problem_desc.strip():
        st.warning("Please enter the problem description.")
    else:
        text = problem_desc + " " + input_desc + " " + output_desc
        X = vectorizer.transform([text])
        X = svd.transform(X)
        pred_class = clf.predict(X)[0]

        st.success("Prediction Complete!")
        st.metric("Predicted Difficulty Class", pred_class.capitalize())
