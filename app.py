
from pathlib import Path
from src.explainability import explain_delay_risk

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from src.propagation import propagate_delay


# ---------------------------------------------------------
# Paths and page setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="ShipMind AI",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 ShipMind AI")
st.caption(
    "Explainable shipment ETA prediction, delay propagation "
    "and scenario analysis"
)

st.info(
    "Demo mode: all shipment records and operational conditions "
    "are synthetic. Predictions and scenario savings are illustrative."
)


# ---------------------------------------------------------
# Load data and trained models
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    eta_model = joblib.load(MODEL_DIR / "eta_model.joblib")
    delay_model = joblib.load(MODEL_DIR / "delay_model.joblib")
    return eta_model, delay_model


required_files = [
    DATA_PATH,
    MODEL_DIR / "eta_model.joblib",
    MODEL_DIR / "delay_model.joblib",
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Required files are missing:\n\n" + "\n".join(missing_files))
    st.stop()

df = load_data()
eta_model, delay_model = load_models()


# ---------------------------------------------------------
# Recreate the original held-out test set.
# This avoids presenting training rows as unseen predictions.
# ---------------------------------------------------------
_, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["is_delayed"],
)

test_df = df.loc[test_idx].copy()

if test_df.empty:
    st.error("No held-out shipments are available.")
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Shipment controls")

shipment_ids = sorted(test_df["shipment_id"].tolist())

selected_id = st.sidebar.selectbox(
    "Select a held-out shipment",
    shipment_ids,
)

shipment = test_df.loc[
    test_df["shipment_id"] == selected_id
].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("What-if scenario")

sea_reduction = st.sidebar.slider(
    "Reduce sea-transit excess duration",
    min_value=0,
    max_value=80,
    value=20,
    step=5,
)

destination_reduction = st.sidebar.slider(
    "Reduce destination-processing excess duration",
    min_value=0,
    max_value=80,
    value=20,
    step=5,
)

st.sidebar.caption(
    "These sliders represent hypothetical percentage reductions "
    "in modeled stage delays, not guaranteed operational savings."
)


alert_threshold = st.sidebar.slider(
    "Delay alert threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.40,
    step=0.05,
    format="%.2f",
    help="Lower thresholds catch more potential delays but create more false alerts.",
)




# ---------------------------------------------------------
# Predict using only the features available in the model.
# Actual stage durations and final outcomes are excluded.
# ---------------------------------------------------------
features = [
    "base_transit_days",
    "weather_sensitivity",
    "departure_delay_hours",
    "weather_severity",
    "congestion_level",
    "planned_transit_days",
    "route_id",
    "origin",
    "destination",
    "customs_risk",
]

X_one = pd.DataFrame([shipment[features].to_dict()])

predicted_eta = max(
    0.0,
    float(eta_model.predict(X_one)[0]),
)

class_labels = list(delay_model.named_steps["model"].classes_)
late_class_index = class_labels.index(1)

late_probability = float(
    delay_model.predict_proba(X_one)[0][late_class_index]
)

risk_label = (
    "Higher modeled risk"
    if late_probability >= alert_threshold
    else "Lower modeled risk"
)


# ---------------------------------------------------------
# Stage propagation and scenario comparison
# ---------------------------------------------------------
baseline = propagate_delay(shipment)

scenario = propagate_delay(
    shipment,
    sea_improvement=sea_reduction / 100,
    destination_improvement=destination_reduction / 100,
)

# Actual values are shown only for transparent retrospective
# evaluation of the synthetic held-out shipment.
actual_eta = float(shipment["actual_transit_days"])
actual_delay = float(shipment["final_delay_hours"])


# ---------------------------------------------------------
# Shipment overview
# ---------------------------------------------------------
st.subheader(f"Shipment overview: {selected_id}")

route_col1, route_col2, route_col3 = st.columns(3)

route_col1.metric("Origin", str(shipment["origin"]))
route_col2.metric("Destination", str(shipment["destination"]))
route_col3.metric("Route ID", str(shipment["route_id"]))

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Predicted transit time",
    f"{predicted_eta:.2f} days",
)

metric2.metric(
    "Actual simulated transit",
    f"{actual_eta:.2f} days",
)

metric3.metric(
    "Delay probability",
    f"{late_probability:.1%}",
)

metric4.metric(
    "Simulated final delay",
    f"{actual_delay:.1f} hours",
)

st.caption(
    "The actual simulated transit and delay are ground-truth values "
    "for retrospective evaluation. They would not be available "
    "when making a real departure-time prediction."
)

st.markdown(f"**Modeled delay-risk category:** {risk_label}")

if late_probability >= 0.5:
    st.warning(
        "The model estimates at least a 50% probability of exceeding "
        "the 24-hour delay threshold."
    )
else:
    st.success(
        "The model estimates less than a 50% probability of exceeding "
        "the 24-hour delay threshold."
    )


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Delay propagation",
    "🧪 What-if simulator",
    "🔎 Shipment details",
])


with tab1:
    st.subheader("How delay accumulates across stages")

    trace_df = pd.DataFrame(baseline["trace"])

    fig = px.bar(
        trace_df,
        x="stage",
        y="new_stage_delay_hours",
        text="new_stage_delay_hours",
        labels={
            "stage": "Transit stage",
            "new_stage_delay_hours": "New delay (hours)",
        },
        title="Additional delay contributed by each stage",
    )

    fig.update_traces(texttemplate="%{text:.2f} h", textposition="outside")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Additional delay (hours)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        trace_df,
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    left.metric(
        "Total delay before buffer",
        f"{baseline['total_delay_before_buffer_hours']:.2f} h",
    )

    right.metric(
        "Schedule buffer",
        f"{baseline['schedule_buffer_hours']:.2f} h",
    )

    st.write(
        "The propagation engine accumulates the modeled delay "
        "contributions and then applies the schedule buffer."
    )


with tab2:
    st.subheader("Test an operational improvement scenario")

    comparison_df = pd.DataFrame({
        "Scenario": [
            "Original simulated outcome",
            "What-if scenario",
        ],
        "ETA (days)": [
            baseline["baseline_actual_eta_days"],
            scenario["scenario_eta_days"],
        ],
        "Delay after buffer (hours)": [
            baseline["delay_after_buffer_hours"],
            scenario["delay_after_buffer_hours"],
        ],
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    saved_hours = (
        baseline["delay_after_buffer_hours"]
        - scenario["delay_after_buffer_hours"]
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Modeled hours saved",
        f"{saved_hours:.2f} h",
    )

    col2.metric(
        "Scenario ETA",
        f"{scenario['scenario_eta_days']:.3f} days",
    )

    comparison_fig = px.bar(
        comparison_df,
        x="Scenario",
        y="Delay after buffer (hours)",
        text="Delay after buffer (hours)",
        title="Delay comparison",
    )

    comparison_fig.update_layout(showlegend=False)
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.warning(
        "This simulator applies assumed percentage reductions to stage "
        "delays. It does not establish that an operational intervention "
        "will produce these savings in reality."
    )


with tab3:
    st.subheader("Shipment operating conditions")

    details = {
        "Shipment ID": shipment["shipment_id"],
        "Route": f"{shipment['origin']} → {shipment['destination']}",
        "Baseline planned transit": f"{shipment['planned_transit_days']:.2f} days",
        "Departure delay": f"{shipment['departure_delay_hours']:.1f} hours",
        "Weather severity": shipment["weather_severity"],
        "Congestion level": shipment["congestion_level"],
        "Customs risk": shipment["customs_risk"],
        "Weather sensitivity": shipment["weather_sensitivity"],
        "Classification ground truth": (
            "Over 24 hours"
            if shipment["is_delayed"] == 1
            else "Within 24 hours"
        ),
    }

    st.json(details)

    st.subheader("Model interpretation")

    st.write(
        f"The ETA model predicts {predicted_eta:.2f} transit days. "
        f"The simulated outcome is {actual_eta:.2f} days. "
        f"The absolute error for this shipment is "
        f"{abs(predicted_eta - actual_eta):.2f} days."
    )

    st.write(
        f"The delay classifier assigns a {late_probability:.1%} "
        "probability to the over-24-hour delay class."
    )

    st.caption(
        "These outputs are model predictions, not causal explanations. "
        "Stage propagation provides a separate, assumption-driven "
        "breakdown of modeled delay contributions."
    )

    
    st.subheader("Why does the model assign this risk?")

    explanation = explain_delay_risk(shipment)

    st.caption(
        "Contributions are measured on the model's log-odds scale. "
        "Positive values increase the predicted delay tendency; "
        "negative values decrease it. These are model associations, "
        "not proof of causation."
    )

    top_features = explanation.head(10).copy()

    top_features["contribution"] = (
        top_features["contribution"].round(3)
    )

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True,
    )

    explanation_chart = px.bar(
        top_features.sort_values("contribution"),
        x="contribution",
        y="feature",
        color="direction",
        orientation="h",
        title="Top feature contributions",
        labels={
            "contribution": "Contribution to log-odds",
            "feature": "Model feature",
            "direction": "Direction",
        },
    )

    st.plotly_chart(
        explanation_chart,
        use_container_width=True,
    )



# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "ShipMind AI | Synthetic-data prototype | "
    "Predictions are experimental and not intended for live logistics decisions."
)
