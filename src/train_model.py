
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
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"
MODEL_DIR = ROOT / "models"

# Only features intended to be available at departure are used.
# Actual outcomes and final stage durations are excluded.
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


def build_pipeline(estimator):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", estimator),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python src/data_generator.py"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [
        ETA_TARGET,
        DELAY_TARGET,
        "shipment_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().all().any():
        raise ValueError(
            "One or more required columns contain only missing values."
        )

    X = df[FEATURES]
    y_eta = df[ETA_TARGET]
    y_delay = df[DELAY_TARGET].astype(int)

    # ---------------------------------------------------------
    # 1. OUTER SPLIT: reserve 20% for final testing.
    # ---------------------------------------------------------
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=y_delay,
    )

    # ---------------------------------------------------------
    # 2. INNER SPLIT: split the remaining 80% into:
    #    60% training and 20% validation of the full dataset.
    # ---------------------------------------------------------
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=42,
        stratify=y_delay.loc[train_val_idx],
    )

    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    X_train_val = X.loc[train_val_idx]
    X_test = X.loc[test_idx]

    eta_train = y_eta.loc[train_idx]
    eta_val = y_eta.loc[val_idx]
    eta_train_val = y_eta.loc[train_val_idx]
    eta_test = y_eta.loc[test_idx]

    delay_train = y_delay.loc[train_idx]
    delay_val = y_delay.loc[val_idx]
    delay_train_val = y_delay.loc[train_val_idx]
    delay_test = y_delay.loc[test_idx]

    print("\n========== DATA SPLIT ==========")
    print(f"Training:   {len(train_idx)} records")
    print(f"Validation: {len(val_idx)} records")
    print(f"Final test: {len(test_idx)} records")
    print(
        "The final test set is reserved for evaluation after "
        "model selection."
    )

    # ---------------------------------------------------------
    # 3. ETA REGRESSION: SELECT USING VALIDATION MAE.
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
    eta_candidates = {}

    for name, estimator in regressors.items():
        model = build_pipeline(estimator)
        model.fit(X_train, eta_train)

        val_predictions = model.predict(X_val)

        val_mae = mean_absolute_error(
            eta_val, val_predictions
        )

        val_rmse = mean_squared_error(
            eta_val, val_predictions
        ) ** 0.5

        val_r2 = r2_score(
            eta_val, val_predictions
        )

        eta_results.append({
            "Model": name,
            "Validation MAE (hours)": val_mae * 24,
            "Validation RMSE (hours)": val_rmse * 24,
            "Validation R2": val_r2,
        })

        eta_candidates[name] = estimator

    eta_results.sort(
        key=lambda result: result["Validation MAE (hours)"]
    )

    best_eta_name = eta_results[0]["Model"]

    print("\n========== ETA VALIDATION RESULTS ==========")
    print(pd.DataFrame(eta_results).to_string(index=False))
    print(f"\nSelected ETA model: {best_eta_name}")

    # Refit the selected estimator using all 80% training + validation
    # records. The test set remains untouched.
    best_eta_model = build_pipeline(
        eta_candidates[best_eta_name]
    )

    best_eta_model.fit(X_train_val, eta_train_val)

    # Final evaluation: performed only after model selection.
    eta_test_predictions = best_eta_model.predict(X_test)

    eta_test_mae = mean_absolute_error(
        eta_test, eta_test_predictions
    )

    eta_test_rmse = mean_squared_error(
        eta_test, eta_test_predictions
    ) ** 0.5

    eta_test_r2 = r2_score(
        eta_test, eta_test_predictions
    )

    joblib.dump(
        best_eta_model,
        MODEL_DIR / "eta_model.joblib",
    )

    print("\n========== FINAL ETA TEST RESULTS ==========")
    print(f"MAE:  {eta_test_mae * 24:.2f} hours")
    print(f"RMSE: {eta_test_rmse * 24:.2f} hours")
    print(f"R2:   {eta_test_r2:.4f}")
    print("Saved: models/eta_model.joblib")

    # ---------------------------------------------------------
    # 4. DELAY CLASSIFICATION: SELECT USING VALIDATION ACCURACY.
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
    delay_candidates = {}

    for name, estimator in classifiers.items():
        model = build_pipeline(estimator)
        model.fit(X_train, delay_train)

        val_predictions = model.predict(X_val)

        val_accuracy = accuracy_score(
            delay_val, val_predictions
        )

        val_f1 = f1_score(
            delay_val, val_predictions, zero_division=0
        )

        delay_results.append({
            "Model": name,
            "Validation accuracy": val_accuracy,
            "Validation F1": val_f1,
        })

        delay_candidates[name] = estimator

    # Preserve accuracy as the selection criterion, but use validation
    # data rather than the final test set.
    delay_results.sort(
        key=lambda result: result["Validation accuracy"],
        reverse=True,
    )

    best_delay_name = delay_results[0]["Model"]

    print("\n========== DELAY VALIDATION RESULTS ==========")
    print(pd.DataFrame(delay_results).to_string(index=False))
    print(f"\nSelected delay model: {best_delay_name}")

    best_delay_model = build_pipeline(
        delay_candidates[best_delay_name]
    )

    best_delay_model.fit(X_train_val, delay_train_val)

    # Final classification evaluation on the untouched test set.
    delay_test_predictions = best_delay_model.predict(X_test)

    delay_test_probabilities = best_delay_model.predict_proba(X_test)

    late_class_index = list(
        best_delay_model.named_steps["model"].classes_
    ).index(1)

    late_probabilities = delay_test_probabilities[:, late_class_index]

    delay_test_accuracy = accuracy_score(
        delay_test, delay_test_predictions
    )

    delay_test_precision = precision_score(
        delay_test, delay_test_predictions, zero_division=0
    )

    delay_test_recall = recall_score(
        delay_test, delay_test_predictions, zero_division=0
    )

    delay_test_f1 = f1_score(
        delay_test, delay_test_predictions, zero_division=0
    )

    joblib.dump(
        best_delay_model,
        MODEL_DIR / "delay_model.joblib",
    )

    print("\n========== FINAL DELAY TEST RESULTS ==========")
    print(f"Accuracy:  {delay_test_accuracy:.3f}")
    print(f"Precision: {delay_test_precision:.3f}")
    print(f"Recall:    {delay_test_recall:.3f}")
    print(f"F1 score:  {delay_test_f1:.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            delay_test,
            delay_test_predictions,
            labels=[0, 1],
            target_names=["Within 24 hours", "Over 24 hours"],
            zero_division=0,
        )
    )

    # Brier score measures probability prediction quality.
    # Lower is better; this is supplementary to classification metrics.
    from sklearn.metrics import brier_score_loss

    print(
        "Brier score:",
        round(
            brier_score_loss(delay_test, late_probabilities),
            4,
        ),
    )

    print("Saved: models/delay_model.joblib")

    # ---------------------------------------------------------
    # 5. SAVE MODEL METADATA.
    # ---------------------------------------------------------
    metadata = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "eta_target": ETA_TARGET,
        "delay_target": DELAY_TARGET,
        "selected_eta_model": best_eta_name,
        "selected_delay_model": best_delay_name,
        "split": {
            "training_records": len(train_idx),
            "validation_records": len(val_idx),
            "test_records": len(test_idx),
            "random_state": 42,
        },
        "eta_test_metrics": {
            "mae_days": float(eta_test_mae),
            "mae_hours": float(eta_test_mae * 24),
            "rmse_days": float(eta_test_rmse),
            "rmse_hours": float(eta_test_rmse * 24),
            "r2": float(eta_test_r2),
        },
        "delay_test_metrics": {
            "accuracy": float(delay_test_accuracy),
            "precision": float(delay_test_precision),
            "recall": float(delay_test_recall),
            "f1": float(delay_test_f1),
            "brier_score": float(
                brier_score_loss(delay_test, late_probabilities)
            ),
        },
        "delay_definition": "final_delay_hours > 24",
        "dataset_type": "synthetic",
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "model_metadata.joblib",
    )

    print("\nSaved: models/model_metadata.joblib")
    print("\nTraining and final evaluation complete.")
    print(
        "Important: all reported test metrics are based on "
        "synthetic shipment data."
    )


if __name__ == "__main__":
    main()
