
import pandas as pd
import pytest

from src.propagation import propagate_delay


def make_shipment():
    """Create one controlled shipment for predictable tests."""

    return pd.Series({
        "shipment_id": "TEST-001",
        "planned_origin_hours": 24.0,
        "actual_origin_hours": 30.0,
        "planned_sea_hours": 120.0,
        "actual_sea_hours": 132.0,
        "planned_destination_hours": 24.0,
        "actual_destination_hours": 30.0,
        "planned_final_hours": 12.0,
        "actual_final_hours": 15.0,
        "planned_transit_days": 7.5,
        "schedule_buffer_hours": 6.0,
        "final_delay_hours": 21.0,
    })


def test_zero_improvement_matches_baseline():
    shipment = make_shipment()

    baseline = propagate_delay(shipment)
    no_change = propagate_delay(
        shipment,
        sea_improvement=0.0,
        destination_improvement=0.0,
    )

    assert (
        baseline["baseline_actual_eta_days"]
        == no_change["scenario_eta_days"]
    )
    assert no_change["hours_saved_vs_original"] == 0.0


def test_improvements_do_not_increase_eta():
    shipment = make_shipment()

    baseline = propagate_delay(shipment)
    improved = propagate_delay(
        shipment,
        sea_improvement=0.20,
        destination_improvement=0.20,
    )

    assert (
        improved["scenario_eta_days"]
        <= baseline["baseline_actual_eta_days"]
    )
    assert improved["hours_saved_vs_original"] >= 0


def test_full_stage_reduction_removes_selected_stage_delays():
    shipment = make_shipment()

    result = propagate_delay(
        shipment,
        sea_improvement=1.0,
        destination_improvement=1.0,
    )

    # Origin and final handling delays remain.
    expected_delay = 6.0 + 3.0
    assert result["total_delay_before_buffer_hours"] == expected_delay


@pytest.mark.parametrize(
    "sea,destination",
    [
        (-0.1, 0.2),
        (1.1, 0.2),
        (0.2, -0.1),
        (0.2, 1.1),
    ],
)
def test_invalid_improvements_raise_error(sea, destination):
    with pytest.raises(ValueError):
        propagate_delay(
            make_shipment(),
            sea_improvement=sea,
            destination_improvement=destination,
        )


def test_schedule_buffer_cannot_make_delay_negative():
    shipment = make_shipment()
    shipment["schedule_buffer_hours"] = 1000.0

    result = propagate_delay(shipment)

    assert result["delay_after_buffer_hours"] == 0.0
    assert result["baseline_delay_after_buffer_hours"] == 0.0
