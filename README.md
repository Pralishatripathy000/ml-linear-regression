<img width="1536" height="1024" alt="ChatGPT Image Jun 28, 2026, 11_59_35 PM" src="https://github.com/user-attachments/assets/cab24edd-89cc-4020-a4f8-390716966f65" />


# 📈 ML Linear Regression

> **An end-to-end implementation of Linear Regression using the California Housing dataset, covering exploratory data analysis, model training, evaluation, visualization, and interpretation.**

---

## 📖 Overview

Linear Regression is one of the most fundamental supervised machine learning algorithms used for predicting continuous numerical values. It models the relationship between independent variables and a target variable by fitting the best possible linear equation that minimizes prediction error.

This project demonstrates the complete machine learning workflow, beginning with data exploration and preprocessing, followed by model training, evaluation, regression diagnostics, and interpretation of feature importance.

This repository is part of my **ML Fundamentals** series, where each project focuses on understanding a core machine learning algorithm through theory, implementation, and practical experimentation.

---

## 📊 Dataset

This project uses the **California Housing Dataset**, a well-known regression dataset provided by **Scikit-learn**.

The dataset contains demographic, geographical, and housing information collected from districts across California. Each record represents one district, while the target variable corresponds to the **median house value** for that district.

### Dataset Summary

| Property       |                              Value |
| -------------- | ---------------------------------: |
| Samples        |                             20,640 |
| Features       |                                  8 |
| Target         | Median House Value (`MedHouseVal`) |
| Missing Values |                               None |
| Problem Type   |                         Regression |

### Features

| Feature        | Description                              |
| -------------- | ---------------------------------------- |
| **MedInc**     | Median income of households              |
| **HouseAge**   | Median age of houses                     |
| **AveRooms**   | Average number of rooms per household    |
| **AveBedrms**  | Average number of bedrooms per household |
| **Population** | Population of the district               |
| **AveOccup**   | Average household occupancy              |
| **Latitude**   | Geographic latitude                      |
| **Longitude**  | Geographic longitude                     |

---

## 🚀 Project Workflow

```text
Dataset
    │
    ▼
Data Exploration
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Data Preprocessing
    │
    ▼
Train-Test Split
    │
    ▼
Linear Regression Model
    │
    ▼
Prediction
    │
    ▼
Model Evaluation
    │
    ▼
Residual Analysis
    │
    ▼
Feature Interpretation
```

---

## 📂 Repository Structure

```text
ml-linear-regression
│
├── data/
├── models/
│   └── linear_regression.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── metrics.csv
│   └── coefficients.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## 📈 Model Evaluation

The trained model was evaluated using multiple regression metrics.

| Metric                         |      Score |
| ------------------------------ | ---------: |
| Mean Absolute Error (MAE)      | **0.5332** |
| Mean Squared Error (MSE)       | **0.5559** |
| Root Mean Squared Error (RMSE) | **0.7456** |
| R² Score                       | **0.5758** |

---

## 📊 Visualizations

### Actual vs Predicted Values

<img width="852" height="677" alt="Screenshot 2026-06-26 225725" src="https://github.com/user-attachments/assets/af6f82cb-4c04-42fb-b2ec-d2b7fce1ca3b" />

---

### Residual Plot


<img width="872" height="677" alt="Screenshot 2026-06-26 225753" src="https://github.com/user-attachments/assets/2de1d43a-3bec-4227-ab95-45cc33fd8f1e" />


---

## 🔍 Key Insights

* The California Housing dataset contains clean numerical data without missing values.
* Median Income (`MedInc`) is the strongest predictor of house values.
* Geographic location also plays a significant role in determining housing prices.
* The model captures the overall trend of the data while maintaining reasonable prediction accuracy.
* Residual analysis indicates that most prediction errors are randomly distributed around zero.

---

## 📚 Learning Outcomes

Through this project, I explored:

* The complete regression workflow
* Exploratory Data Analysis (EDA)
* Feature-target relationships
* Training a Linear Regression model
* Model evaluation using regression metrics
* Residual analysis
* Feature coefficient interpretation
* Model serialization using Joblib

---

## 🤝 Acknowledgements

* California Housing Dataset provided by **Scikit-learn**
* Built as part of my **ML Fundamentals** repository series.



