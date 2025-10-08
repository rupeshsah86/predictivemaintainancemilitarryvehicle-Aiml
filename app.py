from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load("best_predictive_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# For showing accuracy, you can hardcode or load from training results
MODEL_ACCURACY = 0.92  # Example, replace with actual

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_data = {
            'Air temperature [K]': [float(request.form['air_temperature'])],
            'Process temperature [K]': [float(request.form['process_temperature'])],
            'Rotational speed [rpm]': [float(request.form['rotational_speed'])],
            'Torque [Nm]': [float(request.form['torque'])],
            'Tool wear [min]': [float(request.form['tool_wear'])],
            'Product ID': [request.form['product_id']],
            'Type': [request.form['type']],
            'UDI': [int(request.form['udi'])],
            'Failure Type': [request.form.get('failure_type', 'Unknown')]
        }

        input_df = pd.DataFrame(input_data)

        # Preprocess input
        input_preprocessed = preprocessor.transform(input_df)

        # Predict
        prediction_class = model.predict(input_preprocessed)[0]
        prediction_proba = model.predict_proba(input_preprocessed)[0]

        if prediction_class == 1:
            message = f"⚠️ Failure detected! Failure Type: {input_df['Failure Type'][0]}"
        else:
            confidence = prediction_proba[0]  # probability of class 0 (no failure)
            message = f"✅ No failure predicted. Confidence: {confidence*100:.2f}%"

        return render_template('result.html', prediction_message=message, accuracy=MODEL_ACCURACY)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
