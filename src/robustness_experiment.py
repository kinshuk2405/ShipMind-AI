
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.data_quality import inject_data_issues, clean_shipments
from src.train_model import FEATURES, ETA_TARGET, DELAY_TARGET, build_pipeline


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "data/processed/shipments.csv"
OUTPUT = ROOT / "data/quality/robustness_comparison.csv"


def evaluate_eta(train, test, label):
    model = build_pipeline(LinearRegression())
    model.fit(train[FEATURES], train[ETA_TARGET])

    pred = model.predict(test[FEATURES])

    return {
        "Experiment": label,
        "Task": "ETA",
        "MAE_hours": mean_absolute_error(
            test[ETA_TARGET], pred
        ) * 24,
        "RMSE_hours": mean_squared_error(
            test[ETA_TARGET], pred
        ) ** 0.5 * 24,
        "R2": r2_score(test[ETA_TARGET], pred),
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "Brier": None,
    }


def evaluate_delay(train, test, label):
    model = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=42)
    )
    model.fit(train[FEATURES], train[DELAY_TARGET].astype(int))

    pred = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])

    positive_index = list(
        model.named_steps["model"].classes_
    ).index(1)

    delay_prob = probabilities[:, positive_index]
    actual = test[DELAY_TARGET].astype(int)

    return {
        "Experiment": label,
        "Task": "Delay classification",
        "MAE_hours": None,
        "RMSE_hours": None,
        "R2": None,
        "Accuracy": accuracy_score(actual, pred),
        "Precision": precision_score(
            actual, pred, zero_division=0
        ),
        "Recall": recall_score(
            actual, pred, zero_division=0
        ),
        "F1": f1_score(actual, pred, zero_division=0),
        "Brier": brier_score_loss(actual, delay_prob),
    }


def main():
    if not ORIGINAL.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {ORIGINAL}"
        )

    original = pd.read_csv(ORIGINAL)

    required = [
        "shipment_id",
        ETA_TARGET,
        DELAY_TARGET,
        *FEATURES,
    ]

    missing = set(required) - set(original.columns)
    if missing:
        raise ValueError(
            f"Original dataset missing columns: {sorted(missing)}"
        )

    if original["shipment_id"].duplicated().any():
        raise ValueError("Original data contains duplicate IDs.")

    if original[["shipment_id", ETA_TARGET, DELAY_TARGET]].isna().any().any():
        raise ValueError("Shipment IDs or target values are missing.")

    # Split once, before introducing any artificial data issues.
    train_ids, test_ids = train_test_split(
        original["shipment_id"],
        test_size=0.20,
        random_state=42,
        stratify=original[DELAY_TARGET].astype(int),
    )

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    original_train = original[
        original["shipment_id"].isin(train_ids)
    ].copy()

    original_test = original[
        original["shipment_id"].isin(test_ids)
    ].copy()

    if train_ids & test_ids:
        raise ValueError("Training and test IDs overlap.")

    # A: Clean baseline.
    clean_train, _ = clean_shipments(original_train)

    # Create artificial issues in training data only.
    # duplicate_rate=0 keeps one row per shipment for fair comparison.
    noisy_train = inject_data_issues(
        original_train,
        seed=42,
        missing_rate=0.03,
        category_noise_rate=0.04,
        duplicate_rate=0.0,
        invalid_rate=0.01,
    )

    # B: Noisy, uncleaned.
    # Leave missing values, category noise and invalid negatives in place.
    noisy_uncleaned_train = noisy_train.copy()

    # C: The same noisy data after cleaning.
    noisy_clean_train, _ = clean_shipments(noisy_train)

    # Normalize the held-out test features without injecting corruption.
    clean_test, _ = clean_shipments(original_test)

    # Verify row identity and target preservation.
    for name, frame in [
        ("Clean baseline", clean_train),
        ("Noisy uncleaned", noisy_uncleaned_train),
        ("Noisy cleaned", noisy_clean_train),
    ]:
        if set(frame["shipment_id"]) != train_ids:
            raise ValueError(
                f"{name} does not contain the expected training IDs."
            )
        if frame["shipment_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs.")

    if set(clean_test["shipment_id"]) != test_ids:
        raise ValueError("Test IDs changed during preparation.")

    if train_ids & set(clean_test["shipment_id"]):
        raise ValueError("Training/test leakage detected.")

    # Injected issues must not alter the target labels.
    expected_targets = original_train.set_index("shipment_id")[
        [ETA_TARGET, DELAY_TARGET]
    ].sort_index()

    for name, frame in [
        ("Noisy uncleaned", noisy_uncleaned_train),
        ("Noisy cleaned", noisy_clean_train),
    ]:
        actual_targets = frame.set_index("shipment_id")[
            [ETA_TARGET, DELAY_TARGET]
        ].sort_index()

        pd.testing.assert_frame_equal(
            expected_targets,
            actual_targets,
            check_dtype=False,
        )

    # Compare all three conditions on exactly the same test set.
    experiments = [
        ("Clean baseline", clean_train),
        ("Noisy, uncleaned", noisy_uncleaned_train),
        ("Noisy, cleaned", noisy_clean_train),
    ]

    results = []

    for label, train_data in experiments:
        results.append(evaluate_eta(train_data, clean_test, label))
        results.append(evaluate_delay(train_data, clean_test, label))

    comparison = pd.DataFrame(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT, index=False)

    print("\nSHIPMIND AI — THREE-WAY ROBUSTNESS EXPERIMENT")
    print("=" * 58)
    print(f"Training rows per condition: {len(original_train)}")
    print(f"Shared untouched test rows:  {len(clean_test)}")
    print("Duplicate IDs in training conditions: none")
    print("Target labels: verified unchanged")
    print("\nRESULTS")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")
    print(
        "\nLower MAE, RMSE and Brier are better. "
        "Higher R2, accuracy, precision, recall and F1 "
        "are generally better."
    )
    print(
        "\nNote: these results describe an artificial "
        "data-quality experiment, not real-world validation."
    )


if __name__ == "__main__":
    main()
