
import pandas as pd
import pytest

from src.data_quality import (
    inject_data_issues,
    audit_data_quality,
    clean_shipments,
)


@pytest.fixture
def sample_shipments():
    return pd.DataFrame({
        "shipment_id": ["SHP-001", "SHP-002", "SHP-003", "SHP-004"],
        "base_transit_days": [10.0, 12.0, 8.0, 9.0],
        "weather_sensitivity": [0.5, 0.7, 0.4, 0.6],
        "departure_delay_hours": [2.0, 0.0, 3.0, 1.0],
        "weather_severity": [1.0, 2.0, 0.0, 1.0],
        "congestion_level": [2.0, 3.0, 1.0, 2.0],
        "planned_transit_days": [10.0, 12.0, 8.0, 9.0],
        "route_id": ["R1", "R2", "R3", "R4"],
        "origin": ["Singapore", "Mumbai", "Dubai", "Tokyo"],
        "destination": ["Mumbai", "Dubai", "Tokyo", "Singapore"],
        "customs_risk": ["Low", "Medium", "High", "Low"],
        "is_delayed": [0, 1, 0, 1],
    })


def test_injection_is_reproducible(sample_shipments):
    first = inject_data_issues(sample_shipments, seed=42)
    second = inject_data_issues(sample_shipments, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_injection_does_not_modify_original(sample_shipments):
    original = sample_shipments.copy(deep=True)

    inject_data_issues(sample_shipments, seed=42)

    pd.testing.assert_frame_equal(sample_shipments, original)


def test_invalid_rates_are_rejected(sample_shipments):
    with pytest.raises(ValueError):
        inject_data_issues(sample_shipments, missing_rate=1.5)


def test_audit_detects_duplicate_ids(sample_shipments):
    duplicated = pd.concat(
        [sample_shipments, sample_shipments.iloc[[0]]],
        ignore_index=True,
    )

    audit = audit_data_quality(duplicated)
    result = audit.loc[
        audit["Check"] == "Duplicate shipment IDs",
        "Count",
    ].iloc[0]

    assert result == 1


def test_cleaning_removes_duplicate_ids(sample_shipments):
    duplicated = pd.concat(
        [sample_shipments, sample_shipments.iloc[[0]]],
        ignore_index=True,
    )

    cleaned, report = clean_shipments(duplicated)

    assert len(cleaned) == 4
    assert cleaned["shipment_id"].is_unique
    assert report["duplicate_rows_removed"] == 1


def test_cleaning_normalizes_categories(sample_shipments):
    messy = sample_shipments.copy()
    messy.loc[0, "origin"] = "  SINGAPORE  "

    cleaned, _ = clean_shipments(messy)

    assert cleaned.loc[0, "origin"] == "singapore"


def test_cleaning_marks_invalid_values_missing(sample_shipments):
    messy = sample_shipments.copy()
    messy.loc[0, "departure_delay_hours"] = -5

    cleaned, _ = clean_shipments(messy)

    assert pd.isna(cleaned.loc[0, "departure_delay_hours"])


def test_target_labels_are_not_imputed(sample_shipments):
    messy = sample_shipments.copy()
    messy.loc[0, "is_delayed"] = None

    cleaned, _ = clean_shipments(messy)

    assert pd.isna(cleaned.loc[0, "is_delayed"])
