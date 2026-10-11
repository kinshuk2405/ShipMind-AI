
import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    explain_delay_risk,
    explain_delay_risk_shap,
    explain_delay_risk_lime,
)


@pytest.fixture(scope="module")
def shipment():
    """Load a representative shipment from the project dataset."""
    df = pd.read_csv("data/processed/shipments.csv")

    assert not df.empty, "Shipment dataset must not be empty."

    return df.iloc[0]


def assert_valid_explanation(result, method_name):
    """Check the basic structure and numerical validity of an explanation."""
    assert isinstance(result, pd.DataFrame), (
        f"{method_name} must return a pandas DataFrame."
    )
    assert not result.empty, f"{method_name} returned an empty explanation."
    assert len(result.columns) > 0, (
        f"{method_name} must return at least one column."
    )

    numeric_columns = result.select_dtypes(include=np.number)

    assert not numeric_columns.empty, (
        f"{method_name} should contain numerical explanation values."
    )

    assert np.isfinite(numeric_columns.to_numpy()).all(), (
        f"{method_name} contains NaN or infinite numerical values."
    )


def test_coefficient_explanation(shipment):
    result = explain_delay_risk(shipment)
    assert_valid_explanation(result, "Coefficient explanation")


def test_shap_explanation(shipment):
    result = explain_delay_risk_shap(shipment)
    assert_valid_explanation(result, "SHAP explanation")


def test_lime_explanation(shipment):
    result = explain_delay_risk_lime(shipment)
    assert_valid_explanation(result, "LIME explanation")
