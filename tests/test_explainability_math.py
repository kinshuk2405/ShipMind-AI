
import numpy as np
import shap
from src.explainability import (
    explain_delay_risk,
    load_explainer_data,
    transform_shipment,
)


def test_shap_values_reconstruct_logistic_regression_output():
    """
    Verify that SHAP contributions plus the baseline reconstruct
    Logistic Regression's decision_function output (log-odds).
    """
    (
        model,
        preprocessor,
        classifier,
        background,
        feature_names,
    ) = load_explainer_data()

    # Use a real shipment from the project dataset.
    import pandas as pd
    from src.explainability import DATA_PATH, FEATURES

    df = pd.read_csv(DATA_PATH)
    shipment = df.iloc[0]

    transformed = transform_shipment(shipment, preprocessor)

    explainer = shap.LinearExplainer(classifier, background)
    explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    # Support SHAP versions that return per-class values.
    if values.ndim == 3:
        class_index = list(classifier.classes_).index(1)
        values = values[0, :, class_index]

        if base_values.ndim == 2:
            base = base_values[0, class_index]
        else:
            base = base_values.reshape(-1)[class_index]
    else:
        values = values.reshape(-1, len(feature_names))[0]
        base = base_values.reshape(-1)[0]

    reconstructed_output = base + values.sum()

    # Logistic Regression decision_function returns binary log-odds.
    model_output = classifier.decision_function(transformed)[0]

    assert np.isfinite(values).all()
    assert np.isfinite(reconstructed_output)

    assert np.isclose(
        reconstructed_output,
        model_output,
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"SHAP reconstruction ({reconstructed_output}) does not match "
        f"model log-odds ({model_output})."
    )


def test_coefficient_contributions_match_model_coefficients():
    """Verify every reported contribution equals x_i multiplied by beta_i."""
    import pandas as pd

    from src.explainability import DATA_PATH, FEATURES

    (
        model,
        preprocessor,
        classifier,
        background,
        feature_names,
    ) = load_explainer_data()

    df = pd.read_csv(DATA_PATH)
    shipment = df.iloc[0]

    transformed = transform_shipment(shipment, preprocessor)

    expected_contributions = (
        transformed[0] * classifier.coef_[0]
    )

    result = explain_delay_risk(shipment)

    # Compare by feature name because the explanation is sorted by magnitude.
    actual = (
        result.set_index("feature")
        .loc[list(feature_names), "contribution"]
        .to_numpy()
    )

    np.testing.assert_allclose(
        actual,
        expected_contributions,
        atol=1e-8,
        rtol=1e-7,
    )

    expected_directions = np.where(
        expected_contributions > 0,
        "Increases modeled log-odds",
        np.where(
            expected_contributions < 0,
            "Decreases modeled log-odds",
            "Neutral",
        ),
    )

    actual_directions = (
        result.set_index("feature")
        .loc[list(feature_names), "direction"]
        .to_numpy()
    )

    assert np.array_equal(actual_directions, expected_directions)
