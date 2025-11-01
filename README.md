# EMPLOYEE-ATTRITION-PREDICTION
Machine Learning project analyzing Employee Attrition using 7 models and Streamlit dashboard.
# 💼 EMPLOYEE ATTRITION PREDICTION & ANALYTICS DASHBOARD  

### 🔍 Predicting Employee Turnover using Machine Learning and Streamlit  

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)  
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/haarinisk22mid0231/predictive/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📘 Overview  

This project predicts **employee attrition** — whether an employee is likely to leave the organization — using advanced **Machine Learning models** and a powerful **interactive Streamlit dashboard**.  

A total of **seven models** were built and evaluated in the [Kaggle Notebook](https://www.kaggle.com/code/haarinisk22mid0231/predictive/).  
Among them, **Logistic Regression** achieved the best performance and powers the final dashboard.

---

## 📂 Dataset  

- **Dataset:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  
- **Source:** IBM HR Analytics Dataset  
- **Records:** 1470 employees  
- **Features:** 35 attributes  
- **Target Variable:** `Attrition` (`Yes` / `No`)

---

## 🧠 Models Implemented  

| Model | Accuracy | Key Observation |
|--------|-----------|----------------|
| Logistic Regression | ✅ **87%** | Best performing, interpretable |
| Random Forest | 85% | Good, slightly overfit |
| XGBoost | 86% | Stable and robust |
| Decision Tree | 82% | Simple, less generalizable |
| SVM | 84% | Performs well on scaled data |
| KNN | 83% | Sensitive to feature scaling |
| Naive Bayes | 80% | Lower accuracy on numeric data |

> 🏆 **Logistic Regression** is selected for the final Streamlit dashboard.

---

## ⚙️ Methodology  

1. **Data Preprocessing**  
   - Encoded categorical variables with `LabelEncoder`  
   - Scaled numerical features using `StandardScaler`  

2. **Exploratory Data Analysis (EDA)**  
   - Correlation heatmaps, distribution plots, attrition ratio  

3. **Model Training & Evaluation**  
   - Trained 7 ML models using `scikit-learn`  
   - Evaluated using Accuracy, F1-Score, ROC-AUC  

4. **Deployment**  
   - Developed an **interactive Streamlit dashboard** for visualization and prediction  

---

## 📈 Key Insights  

- High attrition observed in employees with **low monthly income** and **short tenure**.  
- **Work-life balance** and **job satisfaction** are crucial in retention.  
- Logistic Regression provided consistent and interpretable results across features.

---

## 💻 Streamlit Dashboard  

### 🧭 Dashboard Overview  
- View dataset and feature distribution  
- Compare attributes against attrition rate  
- Explore feature importance  
- Predict employee attrition interactively  

### 🖥️ Run Locally  

```bash
# Clone the repository
git clone https://github.com/Haarini2164/EMPLOYEE-ATTRITION-PREDICTION.git

# Navigate to folder
cd EMPLOYEE-ATTRITION-PREDICTION

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run logistic_regression_dashboard.py
````

🌐 **Local Dashboard URL:** [http://localhost:8501/](http://localhost:8501/)
📊 **Kaggle Notebook:** [View Predictive Analysis Notebook](https://www.kaggle.com/code/haarinisk22mid0231/predictive/)

---

## 📂 Repository Structure

```
EMPLOYEE-ATTRITION-PREDICTION/
│
├── logistic_regression_dashboard.py       # Streamlit dashboard code
├── predictive.ipynb                       # Kaggle notebook (7 ML models)
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation
```

---

## 📸 Dashboard Preview

*(Add screenshots here once uploaded)*
![Dashboard Screenshot](images/dashboard_1.png)
![Feature Analysis](images/feature_analysis.png)

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
* **Platform:** Jupyter Notebook, Streamlit, Kaggle
* **Deployment:** Local / Streamlit Cloud

---

## 🏆 Results

| Metric   | Score |
| -------- | ----: |
| Accuracy |  0.87 |
| F1 Score | 0.872 |
| ROC-AUC  |  0.93 |

📊 **Feature Importance Visualization:**
![Feature Importance](images/feature_importance.png)

---

## 🚀 Future Enhancements

* Deploy dashboard on **Streamlit Cloud / Hugging Face Spaces**
* Integrate **real-time employee data**
* Add **explainable AI (SHAP)** for interpretability

---

## 👩‍💻 Author

**Haarini S.K.**
Final Year B.E - Computer Science and Engineering
📍 *Employee Attrition Prediction using Machine Learning and Streamlit*

🔗 **GitHub:** [Haarini2164](https://github.com/Haarini2164)
📊 **Kaggle:** [haarinisk22mid0231](https://www.kaggle.com/code/haarinisk22mid0231/predictive/)

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

✨ *“Predict to retain — empowering HR with data-driven insights.”*

