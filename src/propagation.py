
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"


def propagate_delay(
    shipment: pd.Series,
    sea_improvement: float = 0.0,
    destination_improvement: float = 0.0,
) -> dict:
    """
    Compare baseline and simulated journey ETAs.

    Improvements reduce positive excess duration at selected stages.
    The same schedule buffer is applied to both scenarios.

    This is a scenario simulation, not a guarantee of real-world savings.
    """

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

        baseline_stage_delay = max(0.0, actual - planned)
        scenario_stage_delay = baseline_stage_delay

        if stage_name == "Sea transit":
            scenario_stage_delay *= 1 - sea_improvement
        elif stage_name == "Destination processing":
            scenario_stage_delay *= 1 - destination_improvement

        baseline_total_delay += baseline_stage_delay
        scenario_total_delay += scenario_stage_delay

        trace.append({
            "stage": stage_name,
            "planned_hours": round(planned, 2),
            "observed_actual_hours": round(actual, 2),
            "baseline_excess_hours": round(baseline_stage_delay, 2),
            "scenario_excess_hours": round(scenario_stage_delay, 2),
            "scenario_hours_saved": round(
                baseline_stage_delay - scenario_stage_delay, 2
            ),
            "baseline_cumulative_delay_hours": round(
                baseline_total_delay, 2
            ),
            "scenario_cumulative_delay_hours": round(
                scenario_total_delay, 2
            ),
        })

    buffer_hours = float(shipment["schedule_buffer_hours"])
    planned_total_hours = float(shipment["planned_transit_days"]) * 24

    baseline_after_buffer = max(
        0.0, baseline_total_delay - buffer_hours
    )
    scenario_after_buffer = max(
        0.0, scenario_total_delay - buffer_hours
    )

    baseline_eta_hours = planned_total_hours + baseline_after_buffer
    scenario_eta_hours = planned_total_hours + scenario_after_buffer

    # Observed dataset duration, kept separate from simulated ETA.
    observed_actual_eta_hours = (
        float(shipment["actual_transit_days"]) * 24
        if "actual_transit_days" in shipment.index
        else None
    )

    hours_saved = baseline_eta_hours - scenario_eta_hours

    return {
        "shipment_id": shipment["shipment_id"],
        "trace": trace,
        "schedule_buffer_hours": round(buffer_hours, 2),

        "baseline_total_delay_before_buffer_hours": round(
            baseline_total_delay, 2
        ),
        "scenario_total_delay_before_buffer_hours": round(
            scenario_total_delay, 2
        ),

        # Backward-compatible key used by the existing dashboard.
        "total_delay_before_buffer_hours": round(
            scenario_total_delay, 2
        ),

        "baseline_delay_after_buffer_hours": round(
            baseline_after_buffer, 2
        ),
        "delay_after_buffer_hours": round(scenario_after_buffer, 2),

        "baseline_modeled_eta_hours": round(baseline_eta_hours, 2),
        "scenario_eta_hours": round(scenario_eta_hours, 2),
        "observed_actual_eta_hours": (
            round(observed_actual_eta_hours, 2)
            if observed_actual_eta_hours is not None
            else None
),

        "baseline_modeled_eta_days": round(
            baseline_eta_hours / 24, 3
        ),
        "scenario_eta_days": round(scenario_eta_hours / 24, 3),

        # Legacy key: this is a modeled baseline ETA, not observed actual ETA.
        "baseline_actual_eta_days": round(baseline_eta_hours / 24, 3),

        "hours_saved_vs_original": round(hours_saved, 2),
        "sea_improvement": sea_improvement,
        "destination_improvement": destination_improvement,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python src/data_generator.py first."
        )

    df = pd.read_csv(DATA_PATH)

    delayed = df[df["final_delay_hours"] > 0]
    if delayed.empty:
        raise ValueError("No delayed shipments found in the dataset.")

    shipment = delayed.iloc[0]

    baseline = propagate_delay(shipment)
    scenario = propagate_delay(
        shipment,
        sea_improvement=0.20,
        destination_improvement=0.20,
    )

    print(f"\nShipment: {shipment['shipment_id']}")

    
    print("\n--- Stage-by-stage comparison ---")

    baseline_trace = pd.DataFrame(baseline["trace"])
    scenario_trace = pd.DataFrame(scenario["trace"])

    comparison = baseline_trace[
        ["stage", "planned_hours", "baseline_excess_hours"]
    ].merge(
        scenario_trace[["stage", "scenario_excess_hours", "scenario_hours_saved"]],
        on="stage",
    )

    print(comparison.to_string(index=False))



    

    print("\n--- Baseline vs scenario ---")
    if baseline["observed_actual_eta_hours"] is not None:
        print(
            f"Observed dataset journey time: "
            f"{baseline['observed_actual_eta_hours']:.2f} hours"
        )
    else:
        print("Observed dataset journey time: unavailable")

    print(
        f"Baseline modeled ETA: "
        f"{baseline['baseline_modeled_eta_hours']:.2f} hours"
    )
    print(
        f"Scenario modeled ETA: "
        f"{scenario['scenario_eta_hours']:.2f} hours"
    )
    print(
        f"Baseline delay after buffer: "
        f"{baseline['baseline_delay_after_buffer_hours']:.2f} hours"
    )
    print(
        f"Scenario delay after buffer: "
        f"{scenario['delay_after_buffer_hours']:.2f} hours"
    )
    print(
        f"Modeled time saved: "
        f"{scenario['hours_saved_vs_original']:.2f} hours"
    )

    print(
        "\nAssumption: 20% reduction in positive excess durations "
        "at sea transit and destination processing."
    )


if __name__ == "__main__":
    main()
