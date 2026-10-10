
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility: the same seed produces the same dataset.
RANDOM_SEED = 42
NUM_SHIPMENTS = 3000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "shipments.csv"


# Illustrative synthetic route profiles.
# Durations are in days. They are not real-world route estimates.
ROUTES = [
    ("R01", "Shanghai", "Los Angeles", 18, 3.0, 0.7),
    ("R02", "Shanghai", "Rotterdam", 30, 4.0, 0.9),
    ("R03", "Shanghai", "Mumbai", 16, 2.5, 0.7),
    ("R04", "Shanghai", "Singapore", 8, 1.5, 0.4),
    ("R05", "Shanghai", "Sydney", 15, 2.5, 0.6),
    ("R06", "Shenzhen", "Los Angeles", 17, 3.0, 0.8),
    ("R07", "Shenzhen", "Rotterdam", 29, 4.0, 0.9),
    ("R08", "Shenzhen", "Mumbai", 14, 2.5, 0.7),
    ("R09", "Shenzhen", "Singapore", 5, 1.2, 0.4),
    ("R10", "Shenzhen", "Dubai", 13, 2.0, 0.6),
    ("R11", "Singapore", "Rotterdam", 23, 3.0, 0.7),
    ("R12", "Singapore", "Los Angeles", 19, 3.0, 0.7),
    ("R13", "Singapore", "Mumbai", 6, 1.2, 0.4),
    ("R14", "Singapore", "Sydney", 10, 1.8, 0.5),
    ("R15", "Singapore", "Dubai", 11, 2.0, 0.6),
    ("R16", "Mumbai", "Rotterdam", 21, 3.5, 0.8),
    ("R17", "Mumbai", "Dubai", 6, 1.3, 0.5),
    ("R18", "Mumbai", "Singapore", 7, 1.5, 0.5),
    ("R19", "Mumbai", "Los Angeles", 25, 4.0, 0.9),
    ("R20", "Dubai", "Rotterdam", 12, 2.0, 0.5),
    ("R21", "Dubai", "Los Angeles", 22, 3.5, 0.8),
    ("R22", "Dubai", "Singapore", 8, 1.5, 0.5),
    ("R23", "Dubai", "Sydney", 17, 2.5, 0.7),
    ("R24", "Rotterdam", "New York", 12, 2.0, 0.5),
    ("R25", "Rotterdam", "Mumbai", 19, 3.0, 0.7),
    ("R26", "Rotterdam", "Singapore", 22, 3.5, 0.8),
    ("R27", "Los Angeles", "Sydney", 16, 2.5, 0.7),
    ("R28", "Los Angeles", "New York", 7, 1.5, 0.5),
    ("R29", "New York", "Rotterdam", 13, 2.0, 0.5),
    ("R30", "Sydney", "Singapore", 10, 1.8, 0.5),
]

ROUTE_COLUMNS = [
    "route_id",
    "origin",
    "destination",
    "base_transit_days",
    "congestion_mean",
    "weather_sensitivity",
]


def generate_shipments(
    num_shipments: int = NUM_SHIPMENTS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic shipments with linked transit-stage delays."""

    if num_shipments < 1:
        raise ValueError("num_shipments must be at least 1.")

    rng = np.random.default_rng(seed)

    route_df = pd.DataFrame(ROUTES, columns=ROUTE_COLUMNS)
    route_indices = rng.integers(0, len(route_df), size=num_shipments)
    shipments = route_df.iloc[route_indices].reset_index(drop=True).copy()

    shipments.insert(
        0,
        "shipment_id",
        [f"SHP-{i:05d}" for i in range(1, num_shipments + 1)],
    )

    # Known-at-departure operating conditions.
    shipments["departure_delay_hours"] = np.maximum(
        0, rng.normal(3, 5, num_shipments)
    ).round(1)

    shipments["weather_severity"] = rng.beta(
        2, 5, num_shipments
    ).round(3)

    shipments["congestion_level"] = np.clip(
        rng.normal(shipments["congestion_mean"], 1.2),
        0,
        10,
    ).round(2)

    shipments["customs_risk"] = rng.choice(
        ["Low", "Medium", "High"],
        size=num_shipments,
        p=[0.55, 0.30, 0.15],
    )

    customs_risk_hours = shipments["customs_risk"].map(
        {"Low": 3, "Medium": 12, "High": 30}
    ).to_numpy()

    # Split the baseline transit time across dependent stages.
    base_days = shipments["base_transit_days"].to_numpy()

    planned_origin_days = base_days * 0.10
    planned_sea_days = base_days * 0.72
    planned_destination_days = base_days * 0.10
    planned_final_days = base_days * 0.08

    # Random operating disturbances. All delay variables are nonnegative.
    origin_delay_hours = np.maximum(
        0,
        rng.normal(2, 4, num_shipments)
        + shipments["departure_delay_hours"].to_numpy() * 0.25,
    )

    weather_delay_hours = np.maximum(
        0,
        (
            shipments["weather_severity"].to_numpy()
            * shipments["weather_sensitivity"].to_numpy()
            * rng.gamma(2, 8, num_shipments)
        ),
    )

    congestion_delay_hours = np.maximum(
        0,
        (
            shipments["congestion_level"].to_numpy()
            * rng.gamma(2, 1.5, num_shipments)
        ),
    )

    customs_delay_hours = np.maximum(
        0,
        rng.exponential(customs_risk_hours),
    )

    destination_delay_hours = np.maximum(
        0,
        rng.gamma(2, 2, num_shipments),
    )

    final_handling_delay_hours = np.maximum(
        0,
        rng.gamma(1.5, 1.5, num_shipments),
    )

    # Planned stage durations in hours.
    shipments["planned_origin_hours"] = (
        planned_origin_days * 24
    ).round(2)
    shipments["planned_sea_hours"] = (
        planned_sea_days * 24
    ).round(2)
    shipments["planned_destination_hours"] = (
        planned_destination_days * 24
    ).round(2)
    shipments["planned_final_hours"] = (
        planned_final_days * 24
    ).round(2)

    # Actual stage durations = planned duration + stage delay.
    # Weather and congestion affect the sea stage; customs affects
    # destination processing. This creates interpretable dependencies.
    shipments["actual_origin_hours"] = (
        shipments["planned_origin_hours"] + origin_delay_hours
    ).round(2)

    shipments["actual_sea_hours"] = (
        shipments["planned_sea_hours"]
        + weather_delay_hours
        + congestion_delay_hours
    ).round(2)

    shipments["actual_destination_hours"] = (
        shipments["planned_destination_hours"]
        + customs_delay_hours
        + destination_delay_hours
    ).round(2)

    shipments["actual_final_hours"] = (
        shipments["planned_final_hours"] + final_handling_delay_hours
    ).round(2)

    planned_total_hours = (
        shipments["planned_origin_hours"]
        + shipments["planned_sea_hours"]
        + shipments["planned_destination_hours"]
        + shipments["planned_final_hours"]
    )

    actual_total_hours = (
        shipments["actual_origin_hours"]
        + shipments["actual_sea_hours"]
        + shipments["actual_destination_hours"]
        + shipments["actual_final_hours"]
    )

    shipments["planned_transit_days"] = (
        planned_total_hours / 24
    ).round(3)

    shipments["actual_transit_days"] = (
        actual_total_hours / 24
    ).round(3)

    # Target: difference between actual and planned arrival.
    shipments["final_delay_hours"] = (
        actual_total_hours - planned_total_hours
    ).round(2)

    shipments["is_delayed"] = (
        shipments["final_delay_hours"] > 24
    ).astype(int)

    # A buffer absorbs some accumulated delay before it affects final ETA.
    shipments["schedule_buffer_hours"] = rng.choice(
        [0, 6, 12, 24], size=num_shipments, p=[0.15, 0.25, 0.40, 0.20]
    )

    shipments["delay_after_buffer_hours"] = np.maximum(
        0,
        shipments["final_delay_hours"]
        - shipments["schedule_buffer_hours"],
    ).round(2)

    shipments["adjusted_eta_days"] = (
        shipments["planned_transit_days"]
        + shipments["delay_after_buffer_hours"] / 24
    ).round(3)

    # Remove helper columns that are not needed in the final dataset.
    shipments.drop(
        columns=["congestion_mean"],
        inplace=True,
    )

    # Validate core relationships before saving.
    assert len(shipments) == num_shipments
    assert shipments["shipment_id"].is_unique
    assert (shipments["final_delay_hours"] >= 0).all()
    assert shipments[
        ["planned_transit_days", "actual_transit_days"]
    ].notna().all().all()

    return shipments


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_shipments()
    df.to_csv(OUTPUT_PATH, index=False)

    print("ShipMind AI dataset generated successfully!")
    print(f"Rows: {len(df):,}")
    print(f"Unique routes: {df['route_id'].nunique()}")
    print(f"Delayed shipments: {df['is_delayed'].sum():,}")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nSample records:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
