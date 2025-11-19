# **Cancer Prediction using KNN**

A complete end-to-end **Machine Learning project** that uses the **K-Nearest Neighbors (KNN)** algorithm to predict whether a person is likely to have **cancer** based on lifestyle, health, and genetic-risk factors.  
This project includes dataset preprocessing, EDA, model training, evaluation, model saving, and a Streamlit web app for real-time cancer risk prediction.

---

## **Project Overview**

This project classifies **cancer diagnosis (Yes/No)** using the following features:

- age  
- gender  
- bmi  
- smoking  
- genetic_risk  
- physical_activity  
- alcohol_intake  
- cancer_history  
- diagnosis *(Target)*

The dataset undergoes cleaning, label encoding, scaling, and is used to train a **KNN Classification Model**, which is then integrated into a simple **Streamlit UI**.

---

## **Features**

- Load & clean dataset  
- Exploratory Data Analysis (visualizations included)  
- Label encoding of categorical columns  
- Feature scaling using `StandardScaler`  
- Train-test split  
- KNN Training  
- Model Evaluation (Accuracy, Confusion Matrix, Classification Report)  
- Save trained model as `.pkl`  
- Streamlit GUI for predictions  
- Lightweight & production-ready code  

---

## **How the Model Works**

### **1️⃣ Data Preprocessing**
- Handle missing values  
- Label encoding for:  
  - gender  
  - smoking  
  - cancer_history  
  - diagnosis  
- Standardization using **StandardScaler**

### **2️⃣ Model Training**
- Algorithm:  
- Train-test split: **80% training, 20% testing**

### **3️⃣ Model Evaluation**
- Accuracy score  
- Classification report  
- Confusion matrix  
- EDA visualizations (distribution plots, heatmap)

---

## **▶️ Running the Project Locally**

### **Step 1: Install Dependencies**

pip install -r requirements.txt

### **Step 2: Train the Model (Optional)**

python train_cancer_knn.py


This generates:

cancer_knn.pkl

scaler.pkl

### **Step 3: Run the Streamlit App**

streamlit run app.py

### **Deploying on Streamlit Cloud**

-Push all project files to GitHub

-Visit: https://share.streamlit.io

-Click New App

-Select your GitHub repo

-Choose CancerPredictionStreamlit.py as the main app file

-Deploy

### **Technologies Used**

-Python

-Pandas

-NumPy

-Matplotlib

-Seaborn

-Scikit-Learn

-Streamlit

-Joblib
