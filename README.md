# Cancer-Prediction

Cancer Diagnosis Prediction using KNN

A complete end-to-end Machine Learning project that uses the K-Nearest Neighbors (KNN) algorithm to predict whether a person is likely to have cancer based on lifestyle, health, and genetic-risk features.
This project includes EDA, preprocessing, model training, evaluation, model saving, and a Streamlit web application for real-time prediction.

Project Overview

This project aims to classify cancer diagnosis (Yes/No) using the following features:

age

gender

bmi

smoking

genetic_risk

physical_activity

alcohol_intake

cancer_history

diagnosis (Target)

The dataset is cleaned, encoded, scaled, and used to train a KNN Classification Model, which is later integrated with a simple Streamlit UI for user interaction.

Features

✔ Load & clean dataset
✔ Exploratory Data Analysis (visualizations included)
✔ Label encoding of categorical columns
✔ Feature scaling using StandardScaler
✔ Train-test split
✔ KNN Training
✔ Model Evaluation (Accuracy, Confusion Matrix, Class Report)
✔ Save trained model as .pkl
✔ Streamlit GUI for predictions
✔ Simple & production-ready code

How the Model Works
1. Data Preprocessing

Missing value handling

Label encoding for:

gender

smoking

cancer_history

diagnosis

Standardization using StandardScaler

2. Model Training

The model uses:

KNeighborsClassifier(n_neighbors=5)


Trained using 80% of the dataset.

3. Model Evaluation

Accuracy score

Classification report

Confusion matrix

Visual EDA plots (distribution, correlation heatmap)

Running the Project Locally
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train the model (optional)
python train_cancer_knn.py


This generates:

cancer_knn.pkl

scaler.pkl

Step 3: Run the Streamlit App
streamlit run app.py

Deploying on Streamlit Cloud

Push all files to a GitHub repository

Go to https://share.streamlit.io

Click New App

Select your repo

Choose CancerPredictionStreamlit.py as the main app file

Deploy 

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

Streamlit

Joblib
