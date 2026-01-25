import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "mean radius": 14.0, "mean texture": 19.0, "mean perimeter": 90.0, "mean area": 650.0, "mean smoothness": 0.09,
    "mean compactness": 0.1, "mean concavity": 0.09, "mean concave points": 0.05, "mean symmetry": 0.18, "mean fractal dimension": 0.06,
    "radius error": 0.4, "texture error": 1.2, "perimeter error": 2.8, "area error": 40.0, "smoothness error": 0.007,
    "compactness error": 0.025, "concavity error": 0.03, "concave points error": 0.01, "symmetry error": 0.02, "fractal dimension error": 0.003,
    "worst radius": 16.0, "worst texture": 25.0, "worst perimeter": 105.0, "worst area": 800.0, "worst smoothness": 0.13,
    "worst compactness": 0.25, "worst concavity": 0.27, "worst concave points": 0.11, "worst symmetry": 0.29, "worst fractal dimension": 0.08
}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
