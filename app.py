from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your ML models
bloom_model = joblib.load('bloom_model.pkl')  # Replace with your actual model file path
population_model = joblib.load('bloom_model.pkl')  # Replace with your actual model file path

# Endpoint for predicting algal bloom
@app.route('/predict-bloom', methods=['POST'])
def predict_bloom():
    data = request.get_json()
    try:
        # Extract features from input data
        light_intensity = data.get('lightIntensity', 0)
        nitrate = data.get('nitrate', 0)
        features = np.array([[light_intensity, nitrate]])

        # Predict using the bloom model
        prediction = bloom_model.predict(features)[0]
        return jsonify({'result': f'Algal Bloom Prediction: {prediction}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for predicting algae population
@app.route('/predict-population', methods=['POST'])
def predict_population():
    data = request.get_json()
    try:
        # Extract features from input data
        temperature = data.get('temperature', 0)
        co2 = data.get('co2', 0)
        features = np.array([[temperature, co2]])

        # Predict using the population model
        prediction = population_model.predict(features)[0]
        return jsonify({'result': f'Algae Population Prediction: {prediction}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
