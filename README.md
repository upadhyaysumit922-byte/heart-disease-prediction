# ❤️ Heart Disease Prediction

A Machine Learning web app that predicts the likelihood of heart disease based on patient health data.

## 🔗 Live Demo
**Try it here:** https://heart-disease-prediction-1.onrender.com/

## 📌 About
This project uses the Cleveland Heart Disease dataset to train a machine learning model that predicts whether a patient is at risk of heart disease, based on medical attributes like age, blood pressure, cholesterol, and more.

## 🛠️ Tech Stack
- Python
- Scikit-learn (Machine Learning)
- Pandas / NumPy (Data Processing)
- Streamlit (Web App Interface)

## 🚀 How It Works
1. Enter patient health details (age, chest pain type, blood pressure, cholesterol, etc.)
2. Click submit to get a prediction
3. The model returns whether the patient is likely to have heart disease

## 📊 Dataset
- Source: [Cleveland Heart Disease Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- 14 key medical attributes used for prediction

## 📂 Project Files
- `app.py` — Main application file
- `convert.py` — Data preprocessing/conversion script
- `cleveland.data` / `heart.csv` — Dataset files
- `templates/` — HTML templates for the web interface
- `runtime.txt` — Python runtime configuration for deployment

## 💻 Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
