
import pandas as pd
import pytest

from src.explainability import (
    explain_delay_risk,
    explain_delay_risk_shap,
    explain_delay_risk_lime,
)


@pytest.fixture(scope="module")
def shipment():
    """Load one real project sample for explanation tests."""
    df = pd.read_csv("data/processed/shipments.csv")

    assert not df.empty, "Shipment dataset must not be empty."

    return df.iloc[0]


def test_coefficient_explanation(shipment):
    """Verify the existing coefficient-based explanation."""
    result = explain_delay_risk(shipment)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.head(5).shape[0] > 0


def test_shap_explanation(shipment):
    """Verify the SHAP explanation returns usable results."""
    result = explain_delay_risk_shap(shipment)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.head(5).shape[0] > 0


def test_lime_explanation(shipment):
    """Verify the LIME explanation returns usable results."""
    result = explain_delay_risk_lime(shipment)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
