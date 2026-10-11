
from pathlib import Path

import pandas as pd

from src.data_quality import (
    inject_data_issues,
    audit_data_quality,
    clean_shipments,
)


ROOT = Path(__file__).resolve().parent

SOURCE_FILE = ROOT / "data" / "processed" / "shipments.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
QUALITY_DIR = ROOT / "data" / "quality"


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original dataset without modifying it.
    original = pd.read_csv(SOURCE_FILE)

    print("\nSHIPMIND AI — DATA QUALITY PIPELINE")
    print("=" * 45)
    print(f"Original rows: {len(original):,}")
    print(f"Original columns: {len(original.columns)}")

    # Generate a reproducible, separate messy dataset.
    messy = inject_data_issues(
        original,
        seed=42,
        missing_rate=0.03,
        category_noise_rate=0.04,
        duplicate_rate=0.01,
        invalid_rate=0.01,
    )

    messy_path = OUTPUT_DIR / "shipments_messy.csv"
    messy.to_csv(messy_path, index=False)

    # Audit before cleaning.
    audit_before = audit_data_quality(messy)
    audit_before.to_csv(
        QUALITY_DIR / "audit_before.csv",
        index=False,
    )

    print("\n1. QUALITY AUDIT — BEFORE CLEANING")
    print(audit_before.to_string(index=False))

    # Clean the messy dataset.
    cleaned, cleaning_report = clean_shipments(messy)

    cleaned_path = OUTPUT_DIR / "shipments_cleaned.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # Audit after cleaning.
    audit_after = audit_data_quality(cleaned)
    audit_after.to_csv(
        QUALITY_DIR / "audit_after.csv",
        index=False,
    )

    print("\n2. CLEANING SUMMARY")
    for key, value in cleaning_report.items():
        print(f"{key}: {value:,}")

    print("\n3. QUALITY AUDIT — AFTER CLEANING")
    print(audit_after.to_string(index=False))

    # Confirm that the source file was not modified.
    original_after = pd.read_csv(SOURCE_FILE)

    pd.testing.assert_frame_equal(
        original,
        original_after,
        check_dtype=False,
    )

    print("\n4. OUTPUT FILES")
    print(f"Messy dataset:   {messy_path}")
    print(f"Cleaned dataset: {cleaned_path}")
    print(f"Quality reports: {QUALITY_DIR}")

    print("\nOriginal dataset remains unchanged.")
    print("Step 2 completed successfully.")


if __name__ == "__main__":
    main()
