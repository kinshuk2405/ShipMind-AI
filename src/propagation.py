
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"



def propagate_delay(
    shipment: pd.Series,
    sea_improvement: float = 0.0,
    destination_improvement: float = 0.0,
) -> dict:
    """Compare baseline and scenario ETAs using the same schedule buffer."""

    for name, value in {
        "sea_improvement": sea_improvement,
        "destination_improvement": destination_improvement,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    stages = [
        ("Origin handling", "planned_origin_hours", "actual_origin_hours"),
        ("Sea transit", "planned_sea_hours", "actual_sea_hours"),
        (
            "Destination processing",
            "planned_destination_hours",
            "actual_destination_hours",
        ),
        ("Final handling", "planned_final_hours", "actual_final_hours"),
    ]

    baseline_total_delay = 0.0
    scenario_total_delay = 0.0
    trace = []

    for stage_name, planned_col, actual_col in stages:
        planned = float(shipment[planned_col])
        actual = float(shipment[actual_col])
        original_stage_delay = max(0.0, actual - planned)

        adjusted_stage_delay = original_stage_delay

        if stage_name == "Sea transit":
            adjusted_stage_delay *= 1 - sea_improvement
        elif stage_name == "Destination processing":
            adjusted_stage_delay *= 1 - destination_improvement

        trace.append({
            "stage": stage_name,
            "planned_hours": round(planned, 2),
            "new_stage_delay_hours": round(adjusted_stage_delay, 2),
            "inherited_delay_hours": round(scenario_total_delay, 2),
            "cumulative_delay_hours": round(
                scenario_total_delay + adjusted_stage_delay, 2
            ),
        })

        baseline_total_delay += original_stage_delay
        scenario_total_delay += adjusted_stage_delay

    buffer_hours = float(shipment["schedule_buffer_hours"])

    # Apply exactly the same buffer to both cases.
    baseline_after_buffer = max(
        0.0, baseline_total_delay - buffer_hours
    )
    scenario_after_buffer = max(
        0.0, scenario_total_delay - buffer_hours
    )

    planned_total_hours = (
        float(shipment["planned_transit_days"]) * 24
    )

    baseline_eta_days = (
        planned_total_hours + baseline_after_buffer
    ) / 24

    scenario_eta_days = (
        planned_total_hours + scenario_after_buffer
    ) / 24

    return {
        "shipment_id": shipment["shipment_id"],
        "trace": trace,
        "total_delay_before_buffer_hours": round(
            scenario_total_delay, 2
        ),
        "schedule_buffer_hours": round(buffer_hours, 2),
        "delay_after_buffer_hours": round(scenario_after_buffer, 2),
        "baseline_delay_after_buffer_hours": round(
            baseline_after_buffer, 2
        ),
        "baseline_actual_eta_days": round(baseline_eta_days, 3),
        "scenario_eta_days": round(scenario_eta_days, 3),
        "hours_saved_vs_original": round(
            baseline_after_buffer - scenario_after_buffer, 2
        ),
    }



def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python src/data_generator.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # Choose a shipment with a positive delay so the trace is informative.
    delayed = df[df["final_delay_hours"] > 0]
    shipment = delayed.iloc[0]

    baseline = propagate_delay(shipment)
    scenario = propagate_delay(
        shipment,
        sea_improvement=0.20,
        destination_improvement=0.20,
    )

    print(f"\nShipment: {shipment['shipment_id']}")
    print("\n--- Stage-by-stage baseline ---")
    print(pd.DataFrame(baseline["trace"]).to_string(index=False))

    print("\n--- Scenario comparison ---")
    print(f"Original delay: {shipment['final_delay_hours']:.2f} hours")
    print(
        "Scenario delay after buffer: "
        f"{scenario['delay_after_buffer_hours']:.2f} hours"
    )
    print(
        "Modeled hours saved: "
        f"{scenario['hours_saved_vs_original']:.2f}"
    )
    print(
        "Original actual transit time: "
        f"{baseline['baseline_actual_eta_days']:.3f} days"
    )
    print(
        "Scenario ETA: "
        f"{scenario['scenario_eta_days']:.3f} days"
    )
    print(
        "\nScenario assumption: 20% reduction in sea-stage and "
        "destination-stage excess durations."
    )


if __name__ == "__main__":
    main()
