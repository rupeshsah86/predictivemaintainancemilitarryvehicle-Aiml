# Predictive Maintenance Website

A Flask-based web application for predicting equipment maintenance needs using machine learning.

## Features

- Web interface for inputting equipment parameters
- Machine learning model for failure prediction
- Real-time prediction results with confidence scores
- Responsive design with modern UI
- Print functionality for reports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive_maintenance_website
```

2. Install required dependencies:
```bash
pip install flask pandas joblib scikit-learn
```

3. Ensure model files are present:
- `best_predictive_model.pkl`
- `preprocessor.pkl`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Fill in the equipment parameters:
- UDI (Unique Device Identifier)
- Product ID
- Type (L, M, or H)
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (minutes)
- Failure Type

4. Click "Predict Maintenance" to get results

## Project Structure

```
predictive_maintenance_website/
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Input form page
│   └── result.html       # Results display page
├── static/
│   └── style.css         # Styling
├── best_predictive_model.pkl    # Trained ML model
├── preprocessor.pkl             # Data preprocessor
└── README.md
```

## Model Information

- **Algorithm**: Machine Learning Classification
- **Accuracy**: 92%
- **Input Features**: 9 parameters including temperature, speed, torque, and wear
- **Output**: Binary classification (Failure/No Failure)

## API Endpoints

- `GET /` - Home page with input form
- `POST /predict` - Process prediction request

## Requirements

- Python 3.7+
- Flask
- pandas
- joblib
- scikit-learn

## License

© 2023 Predictive Maintenance System. All rights reserved.