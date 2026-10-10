
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "delay_model.joblib"


def explain_delay_risk(shipment: pd.Series) -> pd.DataFrame:
    """Return feature contributions for a Logistic Regression prediction."""

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    if not hasattr(classifier, "coef_"):
        raise TypeError(
            "This explanation function currently supports Logistic Regression."
        )

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

    X = pd.DataFrame([shipment[features].to_dict()])
    transformed = preprocessor.transform(X)

    coefficients = classifier.coef_[0]

    # Use the transformed feature values to estimate each feature's
    # contribution to the model's log-odds for the positive class.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    contributions = transformed[0] * coefficients
    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions,
    })

    explanation["direction"] = explanation["contribution"].apply(
        lambda value: (
            "Increases modeled risk"
            if value > 0
            else "Decreases modeled risk"
            if value < 0
            else "Neutral"
        )
    )

    # Show the largest absolute contributions first.
    explanation["absolute_contribution"] = (
        explanation["contribution"].abs()
    )

    return (
        explanation.sort_values(
            "absolute_contribution",
            ascending=False,
        )
        .drop(columns="absolute_contribution")
        .reset_index(drop=True)
    )
