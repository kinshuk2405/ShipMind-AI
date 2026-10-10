
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"
MODEL_DIR = ROOT / "models"

# Only features available at departure are used.
# Actual stage durations and final outcomes are deliberately excluded.
NUMERIC_FEATURES = [
    "base_transit_days",
    "weather_sensitivity",
    "departure_delay_hours",
    "weather_severity",
    "congestion_level",
    "planned_transit_days",
]

CATEGORICAL_FEATURES = [
    "route_id",
    "origin",
    "destination",
    "customs_risk",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ETA_TARGET = "actual_transit_days"
DELAY_TARGET = "is_delayed"


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python src/data_generator.py"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y_eta = df[ETA_TARGET]
    y_delay = df[DELAY_TARGET]

    # Use identical split indices for both tasks.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=y_delay,
    )

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    eta_train, eta_test = y_eta.loc[train_idx], y_eta.loc[test_idx]
    delay_train, delay_test = (
        y_delay.loc[train_idx],
        y_delay.loc[test_idx],
    )

    # ---------------------------------------------------------
    # 1. ETA regression: compare a baseline with Random Forest.
    # ---------------------------------------------------------
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    eta_results = []
    fitted_regressors = {}

    for name, estimator in regressors.items():
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        model.fit(X_train, eta_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(eta_test, predictions)
        rmse = mean_squared_error(eta_test, predictions) ** 0.5
        r2 = r2_score(eta_test, predictions)

        eta_results.append({
            "model": name,
            "MAE_days": mae,
            "RMSE_days": rmse,
            "R2": r2,
        })
        fitted_regressors[name] = model

    eta_results.sort(key=lambda result: result["MAE_days"])
    best_eta_name = eta_results[0]["model"]
    best_eta_model = fitted_regressors[best_eta_name]

    joblib.dump(
        best_eta_model,
        MODEL_DIR / "eta_model.joblib",
    )

    print("\n========== ETA MODEL RESULTS ==========")
    print(pd.DataFrame(eta_results).to_string(index=False))
    print(f"\nSelected ETA model: {best_eta_name}")
    print("Saved: models/eta_model.joblib")
    print("MAE is the average absolute prediction error in days.")

    # ---------------------------------------------------------
    # 2. Classification: compare Logistic Regression with RF.
    # ---------------------------------------------------------
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    delay_results = []
    fitted_classifiers = {}

    for name, estimator in classifiers.items():
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        model.fit(X_train, delay_train)
        predictions = model.predict(X_test)

        delay_results.append({
            "model": name,
            "accuracy": accuracy_score(delay_test, predictions),
        })
        fitted_classifiers[name] = model

    delay_results.sort(
        key=lambda result: result["accuracy"],
        reverse=True,
    )
    best_delay_name = delay_results[0]["model"]
    best_delay_model = fitted_classifiers[best_delay_name]

    joblib.dump(
        best_delay_model,
        MODEL_DIR / "delay_model.joblib",
    )

    delay_predictions = best_delay_model.predict(X_test)

    print("\n======= DELAY CLASSIFICATION RESULTS =======")
    print(pd.DataFrame(delay_results).to_string(index=False))
    print(f"\nSelected delay model: {best_delay_name}")
    print("\nClassification report:")
    print(classification_report(
        delay_test,
        delay_predictions,
        target_names=["Within 24 hours", "Over 24 hours"],
        zero_division=0,
    ))
    print("Saved: models/delay_model.joblib")

    # Save the feature contract for the dashboard and API.
    joblib.dump(
        {
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "eta_target": ETA_TARGET,
            "delay_target": DELAY_TARGET,
        },
        MODEL_DIR / "model_metadata.joblib",
    )

    print("\nTraining complete!")
    print("Important: these metrics measure performance on synthetic data.")


if __name__ == "__main__":
    main()
