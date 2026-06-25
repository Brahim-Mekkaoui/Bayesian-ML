# 💳 Bayesian Credit Scoring

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-GaussianNB-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Machine%20Learning-Bayesian-green.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Dataset-UCI%20Credit%20Card-blueviolet.svg" alt="Dataset">
</p>

<p align="center">
  <strong>Credit Risk Assessment Using Bayesian Machine Learning</strong>
</p>

---

## 📖 Overview

**Bayesian Credit Scoring** is a machine learning application designed to estimate the probability of credit default using a **Gaussian Naïve Bayes classifier**.

The project is based on the well-known **UCI Credit Card Default Dataset** and provides an interactive **Streamlit dashboard** allowing users to evaluate customer profiles and assess credit risk in real time.

The application demonstrates how Bayesian classification techniques can be applied to financial risk management and credit decision-making.

---

## 🎯 Project Objectives

* Predict whether a customer is likely to default on their credit obligations.
* Demonstrate the application of Bayesian Machine Learning in finance.
* Provide an intuitive interface for risk evaluation.
* Explore credit scoring methodologies using real-world financial data.
* Enable rapid experimentation with customer profiles.

---

## ✨ Features

### 🧠 Machine Learning Model

* Gaussian Naïve Bayes (GaussianNB)
* Automatic model training
* Binary classification (Default / No Default)
* Probability estimation

### 📊 Interactive Dashboard

Built with Streamlit:

* User-friendly interface
* Dynamic financial profile configuration
* Real-time prediction updates
* Risk level visualization

### 📈 Model Evaluation

Performance metrics include:

* Accuracy
* ROC AUC Score
* Default Rate Analysis
* Class Distribution Insights

### 💳 Credit Profile Analysis

The interface allows users to adjust:

* Age
* Gender
* Education Level
* Marital Status
* Credit Limit
* Billing Amounts
* Payment History
* Previous Payment Statuses

---

## 🏗️ Technology Stack

| Layer                | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python 3.8+                     |
| Machine Learning     | Scikit-Learn                    |
| Classification Model | Gaussian Naïve Bayes            |
| Data Processing      | Pandas, NumPy                   |
| Visualization        | Streamlit                       |
| Dataset              | UCI Credit Card Default Dataset |

---

## 📁 Project Structure

```text
brahim-mekkaoui-bayesian-ml/
│
├── credit_scoring_interface.py
├── requirements.txt
├── UCI_Credit_Card.csv
│
└── README.md
```

### Optional Files

```text
ProjetBayesianML.ipynb
```

Contains:

* Exploratory Data Analysis (EDA)
* Feature selection
* Model training
* Performance evaluation

---

## 📊 Dataset

The project uses the:

📌 **Default of Credit Card Clients Dataset**

Key characteristics:

| Attribute | Value                           |
| --------- | ------------------------------- |
| Source    | UCI Machine Learning Repository |
| Records   | 30,000                          |
| Features  | 24                              |
| Target    | Default Payment Next Month      |
| Domain    | Credit Risk                     |

The dataset contains demographic, financial, and repayment information for Taiwanese credit card holders.

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/brahim-mekkaoui-bayesian-ml.git

cd brahim-mekkaoui-bayesian-ml
```

---

### 2️⃣ Create a Virtual Environment

Linux/macOS:

```bash
python -m venv venv

source venv/bin/activate
```

Windows:

```powershell
python -m venv venv

venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download Dataset

Download the dataset from the UCI Machine Learning Repository and place:

```text
UCI_Credit_Card.csv
```

at the project root directory.

Expected structure:

```text
brahim-mekkaoui-bayesian-ml/
│
├── credit_scoring_interface.py
├── requirements.txt
├── UCI_Credit_Card.csv
└── README.md
```

---

## ▶️ Running the Application

Launch Streamlit:

```bash
streamlit run credit_scoring_interface.py
```

Default URL:

```text
http://localhost:8501
```

---

## 💳 How It Works

### Step 1

Configure the customer profile using the sidebar.

### Step 2

Provide financial information:

* Credit limit
* Billing amounts
* Payment history
* Demographic information

### Step 3

Click:

```text
Evaluate Profile
```

### Step 4

Receive:

* Default probability
* Predicted class
* Risk assessment

---

## 📈 Prediction Output

Example result:

| Metric                 | Value     |
| ---------------------- | --------- |
| Probability of Default | 72.4%     |
| Prediction             | Default   |
| Risk Level             | High Risk |

Possible outputs:

### 🟢 Low Risk

```text
Probability < 50%
```

### 🔴 High Risk

```text
Probability ≥ 50%
```

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Feature extraction
* Data cleaning
* Numeric encoding
* Train/Test split

### Model Training

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

### Evaluation

Metrics:

* Accuracy Score
* ROC AUC
* Confusion Matrix
* Classification Report

---

## 🌐 Deployment on Streamlit Cloud

### Push Repository

```bash
git init

git add .

git commit -m "Initial release"

git branch -M main

git remote add origin <repository-url>

git push -u origin main
```

---

### Deploy Application

1. Login to Streamlit Community Cloud
2. Create a new application
3. Select your GitHub repository
4. Choose:

```text
credit_scoring_interface.py
```

as the main file.

5. Deploy.

---

## 🔧 Potential Improvements

Future enhancements may include:

* Model persistence using Joblib
* Advanced feature engineering
* Missing value handling
* Hyperparameter optimization
* Comparison with Logistic Regression
* Comparison with Random Forest
* Explainable AI (SHAP)
* Credit risk dashboards
* REST API integration

---

## 📊 Future Models for Benchmarking

Potential algorithms:

| Model               | Purpose              |
| ------------------- | -------------------- |
| Logistic Regression | Baseline             |
| Random Forest       | Ensemble Learning    |
| XGBoost             | Gradient Boosting    |
| LightGBM            | Large-scale Scoring  |
| CatBoost            | Categorical Features |
| Neural Networks     | Deep Learning        |

---

## 🤝 Contributing

Contributions are welcome.

### Development Workflow

```bash
git checkout -b feature/new-feature

git commit -m "Add new feature"

git push origin feature/new-feature
```

Open a Pull Request for review.

---

## 👨‍💻 Author

**Brahim Mekkaoui**

Machine Learning Student



## 📄 License

This project is intended for educational and research purposes.


---


<p align="center">
💳 Applying Bayesian Machine Learning to Smarter Credit Risk Assessment
</p>
