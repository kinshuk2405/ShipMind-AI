
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"
MODEL_PATH = ROOT / "models" / "delay_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    features = [
        "base_transit_days",
        "weather_sensitivity",
        "departure_delay_hours",
        "weather_severity",
        "congestion_level",
        "planned_transit_days",
        "route_id",
        "origin",
        "destination",
        "customs_risk",
    ]

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df["is_delayed"],
    )

    X_train = df.loc[train_idx, features]
    X_test = df.loc[test_idx, features]

    y_train = df.loc[train_idx, "is_delayed"]
    y_test = df.loc[test_idx, "is_delayed"]

    # Model probabilities and predictions.
    classes = list(model.named_steps["model"].classes_)
    late_index = classes.index(1)

    late_probs = model.predict_proba(X_test)[:, late_index]
    
    print("\n========== ALERT THRESHOLD COMPARISON ==========")

    threshold_rows = []

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        threshold_predictions = (late_probs >= threshold).astype(int)

        threshold_rows.append({
            "threshold": threshold,
            "alerts": int(threshold_predictions.sum()),
            "precision": precision_score(
                y_test, threshold_predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, threshold_predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, threshold_predictions, zero_division=0
            ),
            "missed_delays": int(
                (
                    (y_test.to_numpy() == 1)
                    & (threshold_predictions == 0)
                ).sum()
            ),
        })

    print(
        pd.DataFrame(threshold_rows)
        .round(3)
        .to_string(index=False)
    )

    predictions = (late_probs >= 0.50).astype(int)

    # Baseline: predict the training-set delay rate for every shipment.
    baseline_rate = float(y_train.mean())
    baseline_probs = [baseline_rate] * len(y_test)
    baseline_predictions = [int(baseline_rate >= 0.50)] * len(y_test)

    results = pd.DataFrame([
        {
            "model": "Constant-rate baseline",
            "accuracy": accuracy_score(y_test, baseline_predictions),
            "precision": precision_score(
                y_test, baseline_predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, baseline_predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, baseline_predictions, zero_division=0
            ),
            "brier_score": brier_score_loss(y_test, baseline_probs),
            "log_loss": log_loss(
                y_test, baseline_probs, labels=[0, 1]
            ),
        },
        {
            "model": "Logistic Regression",
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "brier_score": brier_score_loss(y_test, late_probs),
            "log_loss": log_loss(
                y_test, late_probs, labels=[0, 1]
            ),
        },
    ])

    print("\n========== MODEL VS BASELINE ==========")
    print(results.round(4).to_string(index=False))

    print(f"\nTraining-set delay rate: {baseline_rate:.1%}")
    print(f"Test-set delay rate: {y_test.mean():.1%}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Within 24 hours", "Over 24 hours"],
        zero_division=0,
    ))

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("========== CONFUSION MATRIX ==========")
    print(f"Correctly identified on-time shipments: {tn}")
    print(f"False delay alerts: {fp}")
    print(f"Missed delayed shipments: {fn}")
    print(f"Correctly identified delayed shipments: {tp}")

    print("\nInterpretation:")
    print("- Higher precision means fewer false delay alerts.")
    print("- Higher recall means fewer delayed shipments are missed.")
    print("- Higher F1 means a better precision/recall balance.")
    print("- Lower Brier score and log loss indicate better probability predictions.")
    print("- All metrics are measured on synthetic data.")





if __name__ == "__main__":
    main()
