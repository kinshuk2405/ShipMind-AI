
import pandas as pd
import pytest

from src.explainability import (
    FEATURES,
    transform_shipment,
    explain_delay_risk_lime,
)


class IdentityPreprocessor:
    """Minimal preprocessor for testing input validation."""

    def transform(self, frame):
        return frame.to_numpy()


@pytest.fixture
def valid_shipment():
    return pd.Series({
        "base_transit_days": 10.0,
        "weather_sensitivity": 0.5,
        "departure_delay_hours": 2.0,
        "weather_severity": 1.0,
        "congestion_level": 2.0,
        "planned_transit_days": 12.0,
        "route_id": "R001",
        "origin": "Origin A",
        "destination": "Destination B",
        "customs_risk": "Low",
    })


def test_transform_valid_shipment(valid_shipment):
    result = transform_shipment(
        valid_shipment, IdentityPreprocessor()
    )

    assert result.shape == (1, len(FEATURES))


def test_transform_rejects_missing_feature(valid_shipment):
    shipment = valid_shipment.drop("origin")

    with pytest.raises(ValueError, match="missing required features"):
        transform_shipment(shipment, IdentityPreprocessor())


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_transform_rejects_missing_values(
    valid_shipment, missing_value
):
    shipment = valid_shipment.copy()
    shipment["weather_severity"] = missing_value

    with pytest.raises(ValueError, match="contains missing values"):
        transform_shipment(shipment, IdentityPreprocessor())


@pytest.mark.parametrize("invalid_count", [0, -1, 1.5, True])
def test_lime_rejects_invalid_num_features(invalid_count, valid_shipment):
    # Validation should happen before models or artifacts are loaded.
    with pytest.raises(ValueError, match="positive integer"):
        explain_delay_risk_lime(
            valid_shipment, num_features=invalid_count
        )
