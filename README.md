# ShipMind AI

### Explainable Shipment ETA Prediction and Delay Propagation

ShipMind AI is a prototype for predicting shipment transit time, estimating delay risk, tracing delay accumulation across transit stages, and evaluating hypothetical operational improvements.

**Important:** This project currently uses synthetic shipment data. Its metrics demonstrate the prototype's behavior on generated data and should not be interpreted as real-world logistics performance.

## Key Features

- **ETA prediction:** Compares Linear Regression and Random Forest using mean absolute error (MAE), root mean squared error (RMSE), and R².
- **Delay-risk classification:** Compares Logistic Regression and Random Forest to identify shipments predicted to exceed a 24-hour delay threshold.
- **Delay propagation:** Tracks modeled delay contributions across origin handling, sea transit, destination processing, and final handling.
- **What-if simulation:** Estimates the effect of assumed reductions in selected stage delays.
- **Explainability:** Displays Logistic Regression feature contributions for individual delay-risk predictions.
- **Interactive dashboard:** Uses Streamlit and Plotly for shipment exploration and scenario analysis.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Pytest

## Project Structure

```text
Shipmind-AI/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── train_model.py
│   ├── propagation.py
│   ├── explainability.py
│   └── diagnose_risk.py
├── data/
│   └── processed/
│       └── shipments.csv
├── models/
│   ├── eta_model.joblib
│   ├── delay_model.joblib
│   └── model_metadata.joblib
├── tests/
│   └── test_shipmind.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate Data and Train Models

Run these commands from the project root:

```powershell
python src/data_generator.py
python src/train_model.py
```

## Launch the Dashboard

```powershell
streamlit run app.py
```

## Run Automated Tests

```powershell
python -m pytest tests/ -v
```

## Model Evaluation Results

The latest experiment used 3,000 synthetic shipment records, split into training, validation, and test sets.

- **Training set:** 1,800 shipments
- **Validation set:** 600 shipments
- **Final test set:** 600 shipments

Models were selected using validation data. The selected models were then refitted on the combined training and validation data and evaluated on the held-out test set.

### ETA Regression — Linear Regression

| Metric                         | Final Test Result |
| ------------------------------ | ----------------: |
| Mean Absolute Error (MAE)      |       10.78 hours |
| Root Mean Squared Error (RMSE) |       15.87 hours |
| R² Score                       |            0.9906 |

Linear Regression was selected for ETA prediction based on the validation evaluation.

### Delay Classification — Logistic Regression

The classifier predicts whether a shipment will experience a delay exceeding 24 hours.

| Metric      | Final Test Result |
| ----------- | ----------------: |
| Accuracy    |             70.7% |
| Precision   |             71.8% |
| Recall      |             75.4% |
| F1 Score    |             0.736 |
| Brier Score |            0.1972 |

The results indicate that the classifier identifies delayed shipments with a recall of approximately 75.4% on the synthetic test set.

**Evaluation caveat:** These results are based on synthetic shipment data generated for this prototype. They demonstrate performance under simulated conditions and have not been validated against real-world shipment records.

## Limitations

- Route profiles and operational conditions are illustrative, not real shipping statistics.
- The dataset is generated using simplified assumptions and may not represent actual logistics variability.
- The what-if simulator assumes percentage reductions in selected stage delays; it does not establish causal intervention effects.
- Feature contributions explain the trained model's associations, not proven real-world causes.
- Real-world deployment would require historical shipment data, prediction-time feature validation, calibration, monitoring, and further testing.

## Future Improvements

- Validate predictions against real historical shipment records.
- Evaluate performance across routes and shipment types.
- Add prediction intervals and more rigorous probability calibration.
- Incorporate live operational signals when available.
- Evaluate intervention scenarios against operational costs and constraints.
- Add model monitoring and API integration.
