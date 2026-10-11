# ShipMind AI

### Explainable Shipment ETA Prediction & Delay Propagation

**ShipMind AI** is an interactive machine learning prototype that predicts shipment transit times, estimates delay risk, analyzes delay accumulation across transit stages, and explores hypothetical operational improvements through what-if simulation.

The project combines predictive modeling, operational scenario analysis, and explainable AI to demonstrate how shipment risk predictions can be made easier to understand and investigate.

**[🚀 Live Demo](https://shipmind-ai-5evsf9bo3cbkfk2kv9h4ck.streamlit.app/) · [💻 GitHub Repository](https://github.com/kinshuk2405/ShipMind-AI)**

> **Important:** This prototype uses synthetic shipment data. Reported metrics measure performance on generated data and should not be interpreted as evidence of real-world logistics accuracy.

---

## Key Features

- **Shipment ETA Prediction:** Predicts shipment transit duration using machine learning and compares Linear Regression with Random Forest.
- **Delay-Risk Classification:** Estimates the probability that a shipment will experience a delay exceeding 24 hours.
- **Delay Propagation Analysis:** Examines modeled delay contributions across origin handling, sea transit, destination processing, and final handling.
- **What-If Scenario Simulation:** Evaluates hypothetical reductions in selected stage delays and compares the resulting modeled outcomes.
- **Global SHAP Explainability:** Identifies features with the largest average absolute SHAP contributions across a sample of shipments.
- **Local SHAP Explainability:** Shows how individual features push a selected shipment's model output higher or lower.
- **Local LIME Explainability:** Provides a local approximation of the delay classifier's behavior for a selected shipment.
- **Interactive Dashboard:** Explores shipments, model outputs, risk thresholds, evaluation results, and scenario comparisons using Streamlit and Plotly.
- **Automated Testing:** Includes a Pytest suite for core project functionality.

## Technology Stack

| Category          | Technologies      |
| ----------------- | ----------------- |
| Language          | Python            |
| Data processing   | Pandas, NumPy     |
| Machine learning  | Scikit-learn      |
| Explainable AI    | SHAP, LIME        |
| Dashboard         | Streamlit, Plotly |
| Model persistence | Joblib            |
| Testing           | Pytest            |

## System Architecture

The project follows a pipeline that connects shipment data, predictive models, scenario analysis, explainability, and the dashboard.

```text
Synthetic Shipment Data
          |
          v
 Data Preparation
          |
          v
 Feature Preprocessing
          |
          v
   +------+------+
   |             |
   v             v
 ETA Regression  Delay Classification
   |             |
   v             v
 Transit-Time    Delay-Risk Probability
 Prediction      (Delay > 24 Hours)
   |             |
   +------+------+
          |
          v
  SHAP and LIME Explanations
          |
          v
 Delay Propagation and What-If Analysis
          |
          v
   Streamlit Dashboard
```

The predictive models and the stage-based scenario simulator serve different purposes. The model estimates are learned from generated records, while the scenario simulator evaluates outcomes under explicitly assumed stage-delay reductions.

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

## Getting Started

### Prerequisites

- Python compatible with the project's dependencies
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/kinshuk2405/ShipMind-AI.git
cd ShipMind-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

If the repository already contains compatible dataset and model artifacts, launch the application directly:

```bash
streamlit run app.py
```

Streamlit will display the local URL for the dashboard in your terminal.

### 5. Generate data and retrain models (optional)

To regenerate the synthetic dataset and train the models:

```bash
python src/data_generator.py
python src/train_model.py
```

**Note:** Regenerating the dataset or model artifacts may change the reported evaluation results. The metrics below refer to the documented experiment, not necessarily every subsequent training run.

### 6. Run automated tests

```bash
python -m pytest tests/ -v
```

The current reported test status is **8 passing tests**.

## Machine Learning Approach

### ETA Regression

Two regression algorithms are compared:

- Linear Regression
- Random Forest Regressor

The selected ETA model is **Linear Regression**, based on validation-set evaluation.

The regression target is actual shipment transit duration. The model operates on duration values internally, while the dashboard presents ETA durations in hours.

### Delay-Risk Classification

Two classification algorithms are compared:

- Logistic Regression
- Random Forest Classifier

The selected classifier is **Logistic Regression**.

The classification target is whether the final shipment delay exceeds 24 hours:

- **Within threshold:** Delay of 24 hours or less
- **Delayed:** Delay greater than 24 hours

The model estimates a delay probability, which can be compared against a configurable alert threshold.

### Model Evaluation Methodology

The documented experiment uses 3,000 synthetic shipment records:

| Dataset split |   Records |
| ------------- | --------: |
| Training      |     1,800 |
| Validation    |       600 |
| Final test    |       600 |
| **Total**     | **3,000** |

Models are selected using the validation set. The selected pipelines are then refitted on the combined training and validation data and evaluated on the held-out final test set.

This separation helps avoid selecting models using the final test results.

## Model Evaluation Results

### ETA Regression — Linear Regression

| Metric                         | Final test result |
| ------------------------------ | ----------------: |
| Mean Absolute Error (MAE)      |       10.78 hours |
| Root Mean Squared Error (RMSE) |       15.87 hours |
| R² Score                       |            0.9906 |

**Interpretation:** On the synthetic held-out test set, the model's average absolute prediction error is approximately 10.78 hours. The R² score indicates a close fit to the generated target values under the dataset's assumptions.

The high R² score does not establish that the model will achieve comparable accuracy on real shipment data.

### Delay Classification — Logistic Regression

| Metric      | Final test result |
| ----------- | ----------------: |
| Accuracy    |             70.7% |
| Precision   |             71.8% |
| Recall      |             75.4% |
| F1 Score    |             0.736 |
| Brier Score |            0.1972 |

**Interpretation:** At the evaluated classification threshold, recall of 75.4% means the classifier identified approximately three-quarters of the delayed shipments in the synthetic test set.

Precision measures how often predicted delayed shipments were actually delayed in that test set. The Brier score evaluates the accuracy of predicted probabilities, with lower values generally indicating better probabilistic predictions.

These metrics describe the documented test experiment and are not guarantees of production performance.

## Explainable AI: SHAP and LIME

ShipMind AI incorporates explainability to help users investigate how the delay classifier produces its predictions.

### Global SHAP

Global SHAP importance summarizes the average absolute SHAP contribution of each transformed feature across a sample of shipments.

It helps identify which features have the largest influence on the model's output across the analyzed sample.

### Local SHAP

Local SHAP contributions explain an individual shipment's prediction by showing which features push the model output higher or lower relative to its baseline.

Positive and negative contributions describe the direction of influence on the model output. They should not automatically be interpreted as percentage-point changes in delay probability.

### Local LIME

LIME approximates the model's behavior around a selected shipment to provide a local explanation of its prediction.

It offers a complementary perspective to SHAP, although its explanation is an approximation and may depend on the perturbation and sampling settings.

**Important limitation:** SHAP and LIME explain model behavior, not proven real-world causes. Because the dataset is synthetic and the features are transformed during preprocessing, explanations must be interpreted within the prototype's modeling assumptions.

## Delay Propagation and What-If Simulation

The propagation module examines modeled delay contributions across four transit stages:

1. Origin handling
2. Sea transit
3. Destination processing
4. Final handling

The what-if simulator evaluates scenarios in which excess delays at selected stages are reduced by assumed percentages.

For example, a user can investigate how an assumed reduction in sea-stage or destination-stage excess duration changes the simulated shipment outcome.

These simulations support exploratory operational analysis, but they do not prove that a real intervention would produce the same improvement. They are assumption-based scenarios rather than causal estimates.

## Limitations

- **Synthetic data:** Shipment records and operational conditions are generated using simplified assumptions.
- **Limited external validity:** Model performance has not been validated against historical real-world shipment records.
- **Simplified propagation:** Stage-delay calculations may not capture all dependencies, constraints, or interactions in actual logistics networks.
- **Scenario assumptions:** Simulated delay reductions do not establish causal intervention effects.
- **Model explainability:** Feature contributions explain model outputs, not verified causal relationships.
- **Operational readiness:** Production use would require data-quality controls, prediction-time feature validation, external evaluation, monitoring, and appropriate operational safeguards.

## Future Improvements

- Validate the models using historical shipment records.
- Evaluate performance across routes, shipment types, and delay categories.
- Investigate prediction intervals and uncertainty estimation for ETA predictions.
- Improve probability calibration and evaluate alert thresholds against operational requirements.
- Incorporate live operational signals when reliable data becomes available.
- Evaluate scenario improvements against cost, feasibility, and operational constraints.
- Add model monitoring, drift detection, and API integration.

## Project Status

ShipMind AI is a working prototype with a Streamlit dashboard, trained regression and classification models, explainability visualizations, a scenario simulator, and an automated test suite.

The current implementation demonstrates an end-to-end machine learning workflow, from synthetic data generation and model evaluation to interpretable predictions and interactive scenario analysis.

---

**Repository:** [github.com/kinshuk2405/ShipMind-AI](https://github.com/kinshuk2405/ShipMind-AI)  
**Live Application:** [ShipMind AI on Streamlit](https://shipmind-ai-5evsf9bo3cbkfk2kv9h4ck.streamlit.app/)

_Built as a machine learning prototype exploring shipment ETA prediction, delay-risk estimation, delay propagation, and explainable AI._
