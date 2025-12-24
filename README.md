# ⭐ Multimodal Health Diagnosis Using Deep Learning and Machine Learning
---

## 📌 Overview

This project implements a **multimodal disease prediction system** that uses **ECG time-series data** and **patient vitals** to detect heart-related disease.  
Instead of relying on a single data source, the system combines multiple advanced machine learning models:

- **LSTM (RNN) model** for ECG heartbeat classification  
- **Machine Learning model (Random Forest)** for tabular vitals  
- **Fusion Neural Network** that integrates predictions from both models  

A **Streamlit-based web application** allows users to upload an ECG sample and enter health metrics to receive a final disease prediction.

---

## 🧠 Why Multimodal Learning?

In real-world medical diagnosis, doctors consider multiple inputs such as ECG rhythms, blood pressure, cholesterol, and patient history.  
This project mirrors that clinical workflow by fusing different data modalities, resulting in **more reliable and robust predictions**.

---

## 🗂 Project Structure
multimodal_health_ai/
│
├── streamlit_app.py # Streamlit web application
│
├── models/ # Trained models and scalers
│ ├── ecg_rnn_model.keras
│ ├── fusion_model.keras
│ ├── tabular_model.pkl
│ ├── tabular_scaler.pkl
│ ├── ecg_scaler.pkl
│
├── data/
│ ├── heart.csv # Tabular heart disease dataset
│ ├── time_series/
│ │ ├── mitbih_train.csv
│ │ ├── mitbih_test.csv
│ └── fusion/
│ ├── rnn_preds.npy
│ ├── rnn_labels.npy
│ ├── tabular_preds.npy
│ ├── tabular_labels.npy
│
├── notebooks/
│ ├── 1_tabular_model.ipynb
│ ├── 2_ecg_rnn_model.ipynb
│ ├── 3_fusion_model.ipynb
│
└── README.md

---

## 📊 Datasets Used

### 1️⃣ MIT-BIH Arrhythmia Dataset
- ECG heartbeat signals with 187 time-series points per sample  
- Original labels: 0–4  
- Converted to binary classification:
0 → Healthy
1–4 → Disease
- Used for training the **ECG LSTM (RNN) model**

### 2️⃣ Heart Disease / Heart Failure Dataset
- Tabular medical data containing patient vitals
- Features used:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Maximum heart rate
- Oldpeak (ST depression)
- Used for training the **tabular ML model**

---

## 🔧 Models Used

### 🫀 ECG LSTM (RNN) Model
- Processes ECG time-series data
- Architecture:
- LSTM layers
- Dense layers with ReLU
- Sigmoid output
- Output: Probability of heart disease from ECG signal
  
---

### 📋 Tabular Machine Learning Model
- Algorithm: Random Forest
- Input: Patient vitals
- Preprocessing: StandardScaler
- Output:Probability of disease based on vitals
  
---

### 🧠 Fusion Model
- Neural network combining: [ECG_Probability, Tabular_Probability]
- Purpose: Learn how to optimally combine predictions from both models
- Output:Final disease probability
  
---

## 🖥 Streamlit Application

The Streamlit interface allows users to:

1. Upload a **single ECG heartbeat CSV file (187 values)**
2. Enter patient vitals
3. View:
 - ECG model prediction
 - Tabular model prediction
 - Final fusion model diagnosis

---

## 🏁 How to Run the Project

###  Install dependencies
### Run the Streamlit app - streamlit run streamlit_app.py




