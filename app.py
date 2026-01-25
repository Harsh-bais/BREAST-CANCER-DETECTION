from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = "artifacts/best_advanced_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load(MODEL_PATH)

# Feature names explicitly ordered as expected by sklearn's load_breast_cancer
# Use standard sklearn names (mean, error, worst)
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = []
        
        # Ensure ordered input
        for feature in FEATURE_NAMES:
            val = data.get(feature)
            if val is None:
                return jsonify({"error": f"Missing value for {feature}"}), 400
            input_data.append(float(val))
        
        # Create DataFrame to match model training feature names if model tracks them
        # (Pipelines with standard scaler usually handle numpy arrays fine, but DF is safer for feature name consistency)
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Predict
        prediction = model.predict(input_df)[0]
        # Check if predict_proba is available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            # Classes are [0: Malignant, 1: Benign] usually in sklearn
            # Let's double check class mapping. 
            # In sklearn load_breast_cancer: 0 = Malignant, 1 = Benign
            
            # Probability of Malignancy (Class 0)
            malignant_prob = probs[0]
            benign_prob = probs[1]
        else:
            malignant_prob = 1.0 if prediction == 0 else 0.0
            
        result = {
            "prediction": int(prediction),
            "label": "Benign" if prediction == 1 else "Malignant",
            "malignant_probability": float(malignant_prob),
            "benign_probability": float(benign_prob)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
