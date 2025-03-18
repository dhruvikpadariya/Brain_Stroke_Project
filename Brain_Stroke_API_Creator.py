#conda install -c conda-forge flask-cors

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*",}})

model = joblib.load("svc_rbf_model.pkl")
scaler = joblib.load("std_scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)  # Convert input to NumPy array

    # Standardize input using the same scaler
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)

