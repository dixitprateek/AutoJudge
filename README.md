
# 🧠 AutoJudge

## Predicting Programming Problem Difficulty

**AutoJudge** is a machine learning–based system that predicts the **difficulty of programming problems** using only their textual content.
The system performs **both classification and regression** to provide:

* A **difficulty class**: *Easy / Medium / Hard*
* A **numerical difficulty score** on a scale of **0–10**

The project also includes an interactive **Streamlit web application** for real-time predictions.

---

## 📌 Project Overview

Assigning difficulty to programming problems is often subjective and inconsistent across platforms. AutoJudge aims to automate this process using **Natural Language Processing (NLP)** and **Machine Learning** techniques applied to problem statements.

The system:

* Uses **textual descriptions only**
* Does **not rely on metadata** such as submission statistics
* Is designed to be **platform-agnostic**

---

## 📊 Dataset Description

The dataset consists of programming problems with the following textual fields:

* `description` – problem statement
* `input_description` – input format
* `output_description` – output format

Each problem is labeled with:

* `problem_class`: **Easy / Medium / Hard**
* `problem_score`: numerical difficulty score

During preprocessing, all text fields are **concatenated into a single representation**, which is used consistently during training and inference.

---

## 🧠 Approach & Models Used

### 🔹 Text Preprocessing

* Missing values handled
* Text fields concatenated:

  ```
  description + input_description + output_description
  ```
* Standard normalization and cleaning

---

### 🔹 Feature Extraction

* **TF-IDF Vectorization**

  * Unigrams and bigrams
  * Stop-word removal
* **Truncated SVD** (for classification models)

---

## 🔍 Classification Models (Easy / Medium / Hard)

Several models were evaluated:

### 1️⃣ Logistic Regression (TF-IDF + SVD) — **Final Model**

* **Accuracy:** 0.464
* **Macro F1:** **0.448**

Chosen due to:

* Balanced class-wise performance
* Highest Macro F1 score
* Better handling of class imbalance

---

### 2️⃣ Random Forest Classifier (TF-IDF + SVD)

* **Accuracy:** **0.512**
* **Macro F1:** 0.362

Despite higher accuracy, this model strongly favored the *Hard* class and performed poorly on *Easy* and *Medium*, making it unsuitable for deployment.

---

### 3️⃣ Sentence Transformer + Linear SVM

* **Accuracy:** 0.498
* **Macro F1:** **0.469**

Showed strong semantic understanding but struggled to clearly separate *Medium* difficulty problems and was computationally heavier.

---

### ✅ Classification Model Selection

Although Random Forest achieved higher accuracy, **Logistic Regression was selected** because **Macro F1** was prioritized over accuracy due to class imbalance.

---

## 📐 Regression Models (Difficulty Score Prediction)

Regression models were trained using **TF-IDF features** to predict a continuous difficulty score.

| Model                       | MAE      | RMSE     |
| --------------------------- | -------- | -------- |
| Ridge Regression            | 1.77     | 2.11     |
| Random Forest Regressor     | 1.77     | 2.10     |
| Gradient Boosting Regressor | **1.76** | **2.09** |

### ✅ Regression Model Selection

**Ridge Regression** was chosen for deployment due to:

* Lower computational complexity
* Stable performance on high-dimensional text data
* Minimal performance gap compared to ensemble models

---

## 🌐 Web Application (Streamlit)

app link https://autojudge.streamlit.app/

The Streamlit app allows users to:

1. Enter:

   * Problem Description
   * Input Description
   * Output Description
2. Click **Predict Difficulty**
3. View:

   * Predicted difficulty class
   * Predicted difficulty score
   * A warning for borderline cases when predictions disagree

All inputs are internally concatenated to match the training setup.

---

## ▶️ Running the Project Locally

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

The app will open at:

```
http://localhost:8501
```



---

## 🎥 Demo Video

A 2–3 minute demo video showcasing:

* Project overview
* Model approach
* Working web interface

📎 **Demo link:** *(provided in the repository)*

---

## 📝 Notes

* Classification and regression models are trained independently.
* Slight disagreement between predicted class and score is expected for borderline problems.
* Macro F1 score was prioritized over accuracy due to class imbalance.

---

## 👤 Author

**Prateek Dixit**
BS–MS (Economics), IIT Roorkee

---
