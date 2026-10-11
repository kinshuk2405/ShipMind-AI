"""Data quality utilities for ShipMind AI.

All generated quality issues are controlled and reproducible.
The original dataset is never modified.
"""

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "base_transit_days",
    "weather_sensitivity",
    "departure_delay_hours",
    "weather_severity",
    "congestion_level",
    "planned_transit_days",
]

CATEGORICAL_FEATURES = [
    "route_id",
    "origin",
    "destination",
    "customs_risk",
]

NON_NEGATIVE_COLUMNS = [
    "base_transit_days",
    "weather_sensitivity",
    "departure_delay_hours",
    "weather_severity",
    "congestion_level",
    "planned_transit_days",
]


def inject_data_issues(
    df: pd.DataFrame,
    seed: int = 42,
    missing_rate: float = 0.03,
    category_noise_rate: float = 0.04,
    duplicate_rate: float = 0.01,
    invalid_rate: float = 0.01,
) -> pd.DataFrame:
    """Return a separate dataset containing controlled quality issues."""

    rates = [
        missing_rate,
        category_noise_rate,
        duplicate_rate,
        invalid_rate,
    ]
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("All corruption rates must be between 0 and 1.")

    messy = df.copy(deep=True)
    rng = np.random.default_rng(seed)

    # 1. Introduce missing values in model input features only.
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if column not in messy.columns:
            continue

        count = int(len(messy) * missing_rate)
        if count:
            positions = rng.choice(
                messy.index.to_numpy(),
                size=min(count, len(messy)),
                replace=False,
            )
            messy.loc[positions, column] = np.nan

    # 2. Introduce inconsistent category formatting.
    for column in ["origin", "destination", "customs_risk"]:
        if column not in messy.columns:
            continue

        candidates = messy.index[messy[column].notna()].to_numpy()
        count = min(
            int(len(messy) * category_noise_rate),
            len(candidates),
        )

        if count:
            positions = rng.choice(candidates, size=count, replace=False)
            messy.loc[positions, column] = (
                messy.loc[positions, column]
                .astype(str)
                .str.lower()
                .str.strip()
                + " "
            )

    # 3. Introduce invalid negative values in selected input features.
    for column in ["departure_delay_hours", "congestion_level"]:
        if column not in messy.columns:
            continue

        candidates = messy.index[ messy[column].notna() ].to_numpy()
        count = min(int(len(messy) * invalid_rate), len(candidates))

        if count:
            positions = rng.choice(candidates, size=count, replace=False)
            messy.loc[positions, column] = -1.0

    # 4. Append duplicate records with duplicate shipment IDs.
    if "shipment_id" in messy.columns:
        count = min(int(len(messy) * duplicate_rate), len(messy))
        if count:
            duplicate_positions = rng.choice(
                messy.index.to_numpy(),
                size=count,
                replace=False,
            )
            duplicates = messy.loc[duplicate_positions].copy()
            messy = pd.concat(
                [messy, duplicates],
                ignore_index=True,
            )

    return messy.reset_index(drop=True)


def audit_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values, duplicate IDs and invalid input values."""

    results = [
        {
            "Check": "Total rows",
            "Count": len(df),
        },
        {
            "Check": "Duplicate shipment IDs",
            "Count": (
                int(df["shipment_id"].duplicated().sum())
                if "shipment_id" in df.columns
                else 0
            ),
        },
        {
            "Check": "Rows with missing values",
            "Count": int(df.isna().any(axis=1).sum()),
        },
        {
            "Check": "Total missing cells",
            "Count": int(df.isna().sum().sum()),
        },
    ]

    for column in NON_NEGATIVE_COLUMNS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            results.append({
                "Check": f"Invalid negative values: {column}",
                "Count": int((values < 0).sum()),
            })

    return pd.DataFrame(results)


def clean_shipments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize categories, remove duplicate IDs and flag invalid inputs.

    Missing feature values are retained for the existing sklearn
    pipeline to impute using statistics learned from training data.
    Target labels are not imputed or altered.
    """

    cleaned = df.copy(deep=True)
    original_rows = len(cleaned)

    # Keep the first occurrence of each shipment ID.
    if "shipment_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(
            subset=["shipment_id"],
            keep="first",
        )

    # Normalize text consistently; missing values remain missing.
    for column in CATEGORICAL_FEATURES:
         if column in cleaned.columns:
            cleaned[column] = (
             cleaned[column]
            .astype(object)
            .apply(
                lambda value: (
                    np.nan
                    if pd.isna(value) or str(value).strip() == ""
                    else str(value).strip().lower()
                )
            )
        )

    # Convert numeric input features and mark invalid negatives as missing.
    for column in NUMERIC_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    for column in NON_NEGATIVE_COLUMNS:
        if column in cleaned.columns:
            invalid_mask = cleaned[column] < 0
            cleaned.loc[invalid_mask, column] = np.nan

    report = {
        "rows_before": original_rows,
        "rows_after": len(cleaned),
        "duplicate_rows_removed": original_rows - len(cleaned),
        "missing_cells_after_cleaning": int(
            cleaned.isna().sum().sum()
        ),
    }

    return cleaned.reset_index(drop=True), report