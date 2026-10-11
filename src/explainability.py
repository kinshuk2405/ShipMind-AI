from pathlib import Path

import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "delay_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"

FEATURES = [
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


def load_explainer_data():
    """Load model, preprocessing pipeline and background data."""
    model = joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing features: {missing}")

    X = df[FEATURES].copy()
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    if not hasattr(classifier, "coef_"):
        raise TypeError(
            "SHAP/LIME implementation expects Logistic Regression."
        )

    # Transform background data using the saved preprocessing pipeline.
    background = preprocessor.transform(X.sample(
        n=min(100, len(X)), random_state=42
    ))

    if hasattr(background, "toarray"):
        background = background.toarray()

    feature_names = preprocessor.get_feature_names_out()

    return model, preprocessor, classifier, background, feature_names



def transform_shipment(shipment, preprocessor):
    """Validate and transform one shipment using the fitted pipeline."""

    missing_features = [
        feature for feature in FEATURES
        if feature not in shipment.index
    ]

    if missing_features:
        raise ValueError(
            f"Shipment is missing required features: {missing_features}"
        )

    row_data = {feature: shipment[feature] for feature in FEATURES}

    missing_values = [
        feature
        for feature, value in row_data.items()
        if pd.isna(value)
    ]

    if missing_values:
        raise ValueError(
            f"Shipment contains missing values: {missing_values}"
        )

    row = pd.DataFrame([row_data])
    transformed = preprocessor.transform(row)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return transformed



def explain_delay_risk(shipment: pd.Series) -> pd.DataFrame:
    """
    Preserve the original coefficient-based explanation.
    Contributions are relative to zero in transformed feature space,
    not SHAP values or baseline-relative effects.
    """
    _, preprocessor, classifier, _, feature_names = (
        load_explainer_data()
    )

    transformed = transform_shipment(shipment, preprocessor)
    contributions = transformed[0] * classifier.coef_[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions,
    })

    explanation["direction"] = np.where(
        explanation["contribution"] > 0,
        "Increases modeled log-odds",
        np.where(
            explanation["contribution"] < 0,
            "Decreases modeled log-odds",
            "Neutral",
        ),
    )

    explanation["absolute_contribution"] = (
        explanation["contribution"].abs()
    )

    return (
        explanation.sort_values(
            "absolute_contribution", ascending=False
        )
        .drop(columns="absolute_contribution")
        .reset_index(drop=True)
    )


def explain_delay_risk_shap(
    shipment: pd.Series,
) -> pd.DataFrame:
    """Return SHAP contributions for one shipment."""
    _, preprocessor, classifier, background, feature_names = (
        load_explainer_data()
    )

    transformed = transform_shipment(shipment, preprocessor)

    explainer = shap.LinearExplainer(classifier, background)
    shap_values = explainer(transformed)

    values = np.asarray(shap_values.values)

    # Handle binary-class output shapes across SHAP versions.
    if values.ndim == 3:
        class_index = list(classifier.classes_).index(1)
        values = values[0, :, class_index]
    else:
        values = values.reshape(-1, len(feature_names))[0]

    result = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
    })

    result["absolute_value"] = result["shap_value"].abs()

    return (
        result.sort_values("absolute_value", ascending=False)
        .drop(columns="absolute_value")
        .reset_index(drop=True)
    )


def explain_delay_risk_lime(
    shipment: pd.Series,
    num_features: int = 10,
) -> pd.DataFrame:
    """Return a local LIME explanation for one shipment."""
    
    if (
        isinstance(num_features, bool)
        or not isinstance(num_features, int)
        or num_features < 1
    ):
        raise ValueError("num_features must be a positive integer.")

    _, preprocessor, classifier, background, feature_names = (
        load_explainer_data()
    )

    transformed = transform_shipment(shipment, preprocessor)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(background),
        feature_names=list(feature_names),
        class_names=["Within 24 hours", "Over 24 hours"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=transformed[0],
        predict_fn=classifier.predict_proba,
        labels=(list(classifier.classes_).index(1),),
        num_features=min(num_features, len(feature_names)),
    )

    class_index = list(classifier.classes_).index(1)

    result = pd.DataFrame(
        explanation.as_list(label=class_index),
        columns=["feature", "lime_weight"],
    )

    result["direction"] = np.where(
        result["lime_weight"] > 0,
        "Supports higher delay risk locally",
        "Supports lower delay risk locally",
    )

    return result

def get_global_shap_importance(
    sample_size: int = 200,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across shipments."""
    _, preprocessor, classifier, background, feature_names = (
        load_explainer_data()
    )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].sample(
        n=min(sample_size, len(df)),
        random_state=42,
    )

    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.LinearExplainer(classifier, background)
    values = explainer(transformed).values
    values = np.asarray(values)

    if values.ndim == 3:
        class_index = list(classifier.classes_).index(1)
        values = values[:, :, class_index]

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(values).mean(axis=0),
    })

    return importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)