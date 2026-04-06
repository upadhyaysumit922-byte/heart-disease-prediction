from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model  = pickle.load(open('model.pkl',  'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f]) for f in [
        'age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope','ca','thal'
    ]]
    data = scaler.transform([features])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        result = "❤️ Heart Disease Detected"
        color = "red"
    else:
        result = "✅ No Heart Disease"
        color = "green"

    return render_template('index.html', result=result, color=color, probability=f"{probability:.1f}")

if __name__ == '__main__':
    app.run(debug=True)