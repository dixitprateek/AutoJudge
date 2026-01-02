# AutoJudge
AutoJudge For Predicting Programming  Problem Difficulty 
Perfect. Below is a **complete, submission-ready `README.md`** tailored **exactly** to your project, models, metrics, and Streamlit app.

You can **copy–paste this directly** into `README.md` in your repo.

---

# 🧠 AutoJudge

### Predicting Programming Problem Difficulty

AutoJudge is a machine learning–based system that predicts the **difficulty level** of programming problems.
Given a problem’s **description, input format, and output format**, the system:

* **Classifies** the problem into **Easy / Medium / Hard**
* **Predicts a continuous difficulty score** on a scale of **0–10**
* Provides results through an interactive **Streamlit web interface**

---

## 📌 Project Overview

Judging the difficulty of programming problems is important for competitive programming platforms, educational tools, and adaptive learning systems.
AutoJudge automates this process using **Natural Language Processing (NLP)** and **Machine Learning** techniques applied to problem statements.

The project consists of:

* A **classification model** for difficulty labels
* A **regression model** for difficulty score prediction
* A **web UI** for real-time inference

---

## 📊 Dataset Used

The dataset consists of programming problems with the following textual fields:

* `description` – problem statement
* `input_description` – input format
* `output_description` – output format

Each problem is annotated with:

* A **difficulty class** (`easy`, `medium`, `hard`)
* A **numeric difficulty score**

During preprocessing, all textual fields are **concatenated into a single text representation**.

---

## 🧠 Approach & Models Used

### 🔹 Text Preprocessing

* Missing values handled
* Text fields concatenated:

  ```
  description + input_description + output_description
  ```
* Standard NLP preprocessing via TF-IDF

---

### 🔹 Feature Extraction

* **TF-IDF Vectorization**
* **Truncated SVD** (for dimensionality reduction in classification)

---

### 🔹 Classification Model

* **Model:** Logistic Regression (Linear classifier)
* **Features:** TF-IDF → Truncated SVD
* **Output:** Easy / Medium / Hard
* **Reason:** Balanced performance across classes (better Macro F1 than tree-based models)

---

### 🔹 Regression Model

* **Model:** Ridge Regression
* **Features:** TF-IDF
* **Output:** Difficulty score (0–10)
* **Reason:** Stable and effective for high-dimensional text features

---

## 📈 Evaluation Metrics

### 🔹 Classification

* Accuracy
* Confusion Matrix
* **Macro F1 Score** (used due to class imbalance)

### 🔹 Regression

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

> Macro F1 was preferred over accuracy to ensure fair performance across all difficulty classes.

---

## 🌐 Web Interface (Streamlit)

The Streamlit application allows users to:

1. Enter:

   * Problem Description
   * Input Description
   * Output Description
2. Click **Predict Difficulty**
3. View:

   * Predicted difficulty class
   * Predicted difficulty score
   * A warning if classification and score indicate borderline difficulty

The UI combines all text inputs internally, consistent with the training setup.

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dixitprateek/autojudge.git
cd autojudge
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 🎥 Demo Video

A 2–3 minute demo video explaining:

* Project overview
* Model approach
* Working web interface

📎 **Demo Link:**
(added in `demo/demo_video_link.txt`)

---

## 📁 Repository Structure

```
autojudge/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── vectorizer.pkl
│   ├── svd.pkl
│   ├── lr_model.pkl
│   ├── tfidf_reg.pkl
│   └── ridge_reg.pkl
│
├── report/
│   └── report.pdf
│
└── demo/
    └── demo_video_link.txt
```

---

## 🧩 Notes on Model Outputs

Classification and regression models are trained independently.
In some cases, the predicted class and score may differ slightly due to ambiguity in problem difficulty.
Such cases are highlighted in the UI as **borderline difficulty**, ensuring transparency.

---

## 👤 Author

**Prateek Dixit**
BS-MS (Economics), IIT Roorkee
Project: *AutoJudge – Predicting Programming Problem Difficulty*

---
