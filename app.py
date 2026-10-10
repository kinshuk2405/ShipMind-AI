from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from src.explainability import explain_delay_risk
from src.propagation import propagate_delay


# =========================================================
# 1. PAGE SETUP
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "shipments.csv"
MODEL_DIR = ROOT / "models"

FEATURES = [
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

TARGET_DELAY = "is_delayed"

st.set_page_config(
    page_title="ShipMind AI",
    page_icon="🚢",
    layout="wide",
)


# =========================================================
# 2. LIGHT / DARK THEME
# =========================================================

theme = st.sidebar.radio(
    "🎨 Appearance",
    options=["☀️ Light", "🌙 Dark"],
    horizontal=True,
)

is_dark = theme == "🌙 Dark"
plotly_template = "plotly_dark" if is_dark else "plotly_white"

if is_dark:
    background = "#111827"
    sidebar_background = "#0B1220"
    card_background = "#1F2937"
    text_color = "#F9FAFB"
    muted_color = "#9CA3AF"
    accent_color = "#67E8F9"
    border_color = "#374151"
else:
    background = "#F4F7FB"
    sidebar_background = "#FFFFFF"
    card_background = "#FFFFFF"
    text_color = "#172B4D"
    muted_color = "#64748B"
    accent_color = "#2563EB"
    border_color = "#E2E8F0"

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {background};
        color: {text_color};
    }}

    [data-testid="stHeader"] {{
        background-color: {background};
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_background};
        border-right: 1px solid {border_color};
    }}

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] li {{
        color: {text_color};
    }}

    [data-testid="stMetric"] {{
        background-color: {card_background};
        border: 1px solid {border_color};
        padding: 18px;
        border-radius: 14px;
        transition: border-color 0.2s ease,
                    transform 0.2s ease;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: {accent_color};
        transform: translateY(-2px);
    }}

    [data-testid="stMetricLabel"] {{
        color: {muted_color} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {accent_color} !important;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {border_color};
        border-radius: 12px;
    }}

    div.stButton > button {{
        border-radius: 10px;
        transition: all 0.2s ease;
    }}

    div.stButton > button:hover {{
        border-color: {accent_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. INTRODUCTION
# =========================================================

st.title("🚢 ShipMind AI")
st.subheader("Understand a shipment at a glance")

st.write(
    "Check the estimated journey duration, the chance of a long delay, "
    "and how hypothetical changes could affect extra time."
)

st.info(
    "Demo only: this project uses simulated shipment data, not live "
    "shipping records. Predictions and scenario results are illustrative."
)


# =========================================================
# 4. LOAD DATA AND MODELS
# =========================================================

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

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:
    st.error(
        "Some project files are missing:\n\n"
        + "\n".join(missing_files)
    )
    st.stop()

df = load_data()
eta_model, delay_model = load_models()

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET_DELAY],
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

if train_df.empty or test_df.empty:
    st.error("Training or testing data is empty.")
    st.stop()


# =========================================================
# 5. SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("1. Choose a shipment")

shipment_ids = sorted(test_df["shipment_id"].tolist())

selected_id = st.sidebar.selectbox(
    "Shipment ID",
    shipment_ids,
)

shipment = test_df.loc[
    test_df["shipment_id"] == selected_id
].iloc[0]

st.sidebar.divider()

st.sidebar.header("2. Try a what-if scenario")

st.sidebar.caption(
    "Imagine reducing the extra time spent at certain stages. "
    "These sliders explore hypothetical improvements."
)

sea_reduction = st.sidebar.slider(
    "Reduce extra time during sea travel",
    min_value=0,
    max_value=80,
    value=20,
    step=5,
    format="%d%%",
)

destination_reduction = st.sidebar.slider(
    "Reduce extra time at destination",
    min_value=0,
    max_value=80,
    value=20,
    step=5,
    format="%d%%",
)

st.sidebar.divider()

alert_threshold = st.sidebar.slider(
    "Flag a shipment when delay chance reaches",
    min_value=0.20,
    max_value=0.80,
    value=0.40,
    step=0.05,
    format="%.0f%%",
    help=(
        "Lower settings flag more shipments. This may catch more delays, "
        "but can also create more false alarms."
    ),
)


# =========================================================
# 6. PREDICTIONS
# =========================================================

X_one = pd.DataFrame([
    shipment[FEATURES].to_dict()
])

# Keep the model's original output in days internally.
predicted_eta_days = max(
    0.0,
    float(eta_model.predict(X_one)[0]),
)

# Convert to hours for display only.
predicted_eta_hours = predicted_eta_days * 24

delay_classes = list(
    delay_model.named_steps["model"].classes_
)

late_class_index = delay_classes.index(1)

delay_probability = float(
    delay_model.predict_proba(X_one)[0][late_class_index]
)

flagged = delay_probability >= alert_threshold

baseline = propagate_delay(shipment)

scenario = propagate_delay(
    shipment,
    sea_improvement=sea_reduction / 100,
    destination_improvement=destination_reduction / 100,
)


# =========================================================
# 7. SHIPMENT SUMMARY
# =========================================================

st.markdown("## 1. Shipment summary")

st.caption(
    f"{shipment['origin']} → {shipment['destination']} "
    f"· Shipment ID: {selected_id}"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Estimated journey duration",
    f"{predicted_eta_hours:.1f} hours",
    help="The model's estimate of the total journey duration.",
)

col2.metric(
    "Chance of delay over 24 hours",
    f"{delay_probability:.0%}",
    help=(
        "Estimated probability that the final shipment delay "
        "will exceed 24 hours."
    ),
)

col3.metric(
    "Attention status",
    "Review recommended" if flagged else "Below alert setting",
)

if flagged:
    st.warning(
        f"The estimated delay chance ({delay_probability:.0%}) "
        f"meets or exceeds your alert setting "
        f"({alert_threshold:.0%})."
    )
else:
    st.success(
        f"The estimated delay chance is below your alert setting "
        f"({alert_threshold:.0%}). This does not guarantee on-time arrival."
    )


with st.expander("Compare prediction with simulated outcome"):

    st.write(
        "This comparison is possible because the demo includes simulated "
        "outcomes. For a real shipment, the final journey duration would "
        "not be known when the initial prediction is made."
    )

    actual_days = float(shipment["actual_transit_days"])
    actual_hours = actual_days * 24
    actual_delay_hours = float(shipment["final_delay_hours"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Simulated actual journey",
        f"{actual_hours:.1f} hours",
    )

    col2.metric(
        "Absolute prediction difference",
        f"{abs(predicted_eta_days - actual_days) * 24:.1f} hours",
    )

    col3.metric(
        "Simulated final delay",
        f"{actual_delay_hours:.1f} hours",
    )


# =========================================================
# 8. STAGE-LEVEL DELAY BREAKDOWN
# =========================================================

st.divider()
st.markdown("## 2. Where does extra time build up?")

st.write(
    "Each point represents extra time assigned to a stage in this "
    "simulated shipment. All values are shown in hours. This breakdown "
    "does not prove what caused a real-world delay."
)

trace = pd.DataFrame(baseline["trace"])

stage_names = {
    "Origin handling": "Starting-point handling",
    "Sea transit": "Sea travel",
    "Destination processing": "Destination processing",
    "Final handling": "Final hand-off",
}

trace["Stage"] = (
    trace["stage"]
    .map(stage_names)
    .fillna(trace["stage"])
)

trace["Extra time (hours)"] = trace["new_stage_delay_hours"]

stage_fig = px.scatter(
    trace,
    x="Extra time (hours)",
    y="Stage",
    size="Extra time (hours)",
    text="Extra time (hours)",
    title="Extra time by shipment stage",
    labels={
        "Stage": "",
        "Extra time (hours)": "Extra time (hours)",
    },
    template=plotly_template,
)

stage_fig.update_traces(
    texttemplate="%{text:.1f} h",
    textposition="middle right",
)

stage_fig.update_layout(
    showlegend=False,
    margin={"l": 10, "r": 40, "t": 45, "b": 10},
)

st.plotly_chart(stage_fig, use_container_width=True)

col1, col2 = st.columns(2)

col1.metric(
    "Schedule allowance",
    f"{baseline['schedule_buffer_hours']:.1f} hours",
    help="Extra time planned into the schedule to absorb some delays.",
)

col2.metric(
    "Extra time remaining after allowance",
    f"{baseline['baseline_delay_after_buffer_hours']:.1f} hours",
    help="Modelled delay remaining after applying the schedule allowance.",
)


# =========================================================
# 9. WHAT-IF SCENARIO
# =========================================================

st.divider()
st.markdown("## 3. What if we reduced extra time?")

st.write(
    "The sliders apply assumed percentage reductions to extra time "
    "during sea travel and destination processing."
)

modeled_hours_saved = max(
    0.0,
    baseline["baseline_delay_after_buffer_hours"]
    - scenario["delay_after_buffer_hours"],
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Before changes",
    f"{baseline['baseline_actual_eta_days'] * 24:.1f} hours",
)

col2.metric(
    "After scenario changes",
    f"{scenario['scenario_eta_days'] * 24:.1f} hours",
)

col3.metric(
    "Extra time reduced",
    f"{modeled_hours_saved:.1f} hours",
)

scenario_df = pd.DataFrame({
    "Scenario": ["Before changes", "After changes"],
    "Extra time after allowance (hours)": [
        baseline["baseline_delay_after_buffer_hours"],
        scenario["delay_after_buffer_hours"],
    ],
})

scenario_fig = px.line(
    scenario_df,
    x="Scenario",
    y="Extra time after allowance (hours)",
    markers=True,
    text="Extra time after allowance (hours)",
    title="Extra time before and after hypothetical changes",
    labels={
        "Scenario": "",
        "Extra time after allowance (hours)": "Extra time (hours)",
    },
    template=plotly_template,
)

scenario_fig.update_traces(
    texttemplate="%{text:.1f} h",
    textposition="top center",
    line={"width": 3},
    marker={"size": 12},
)

scenario_fig.update_layout(
    showlegend=False,
    margin={"l": 10, "r": 10, "t": 45, "b": 10},
)

st.plotly_chart(scenario_fig, use_container_width=True)

st.caption(
    "This is a simplified simulation. It assumes the selected reductions "
    "happen; it does not prove that a real-world action would achieve "
    "the same result."
)


# =========================================================
# 10. MODEL EXPLANATION
# =========================================================

st.divider()

st.markdown("## 4. Which information influenced the risk estimate?")

st.write(
    "This chart shows signals that pushed the model's estimate up or down. "
    "The values are on a model influence scale, not percentages, and do "
    "not prove that a signal caused a delay."
)

explanation = explain_delay_risk(shipment).head(8).copy()


def friendly_name(name):
    label = (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
    )

    replacements = {
        "departure delay hours": "Delay before departure",
        "weather severity": "Weather severity (simulated level)",
        "congestion level": "Congestion (simulated level)",
        "customs risk": "Customs risk (simulated setting)",
        "weather sensitivity": "Route sensitivity to weather",
        "planned transit days": "Planned journey duration feature",
        "base transit days": "Typical journey duration feature",
        "route id": "Route identifier",
        "origin": "Starting point",
        "destination": "Destination",
    }

    for raw, friendly in replacements.items():
        label = label.replace(raw, friendly)

    return label


explanation["Information signal"] = (
    explanation["feature"].map(friendly_name)
)

explanation["Direction"] = explanation["contribution"].apply(
    lambda value: (
        "Pushes risk estimate up"
        if value > 0
        else (
            "Pushes risk estimate down"
            if value < 0
            else "Little influence"
        )
    )
)

explanation = explanation.sort_values("contribution")

influence_fig = px.scatter(
    explanation,
    x="contribution",
    y="Information signal",
    color="Direction",
    size=explanation["contribution"].abs(),
    title="Signals influencing the model's risk estimate",
    labels={
        "contribution": "Model influence (relative scale)",
        "Information signal": "",
    },
    template=plotly_template,
)

influence_fig.update_layout(
    margin={"l": 10, "r": 10, "t": 45, "b": 10},
)

st.plotly_chart(influence_fig, use_container_width=True)

st.caption(
    "The influence scale is not a percentage. These are associations "
    "learned from synthetic data, not proof of real-world causes."
)


# =========================================================
# 11. TRAINING VS TESTING COMPARISON
# =========================================================

st.divider()

st.markdown(
    "## 5. How does the model perform on training vs testing data?"
)

st.write(
    "**Training data** is used to fit the model. **Testing data** is held "
    "back to evaluate performance on examples the model did not train on. "
    "Testing performance is a better indicator of generalization, but "
    "synthetic data cannot establish real-world performance."
)

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_eta_train = train_df["actual_transit_days"]
y_eta_test = test_df["actual_transit_days"]

y_delay_train = train_df[TARGET_DELAY].astype(int)
y_delay_test = test_df[TARGET_DELAY].astype(int)

eta_train_pred = eta_model.predict(X_train)
eta_test_pred = eta_model.predict(X_test)

delay_train_pred = delay_model.predict(X_train).astype(int)
delay_test_pred = delay_model.predict(X_test).astype(int)

performance = pd.DataFrame([
    {
        "Dataset": "Training",
        "ETA MAE (hours)": mean_absolute_error(
            y_eta_train, eta_train_pred
        ) * 24,
        "Accuracy": accuracy_score(
            y_delay_train, delay_train_pred
        ),
        "Precision": precision_score(
            y_delay_train, delay_train_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_delay_train, delay_train_pred, zero_division=0
        ),
        "F1 score": f1_score(
            y_delay_train, delay_train_pred, zero_division=0
        ),
        "Delay rate": y_delay_train.mean(),
        "Average planned journey (hours)": (
            train_df["planned_transit_days"].mean() * 24
        ),
    },
    {
        "Dataset": "Testing",
        "ETA MAE (hours)": mean_absolute_error(
            y_eta_test, eta_test_pred
        ) * 24,
        "Accuracy": accuracy_score(
            y_delay_test, delay_test_pred
        ),
        "Precision": precision_score(
            y_delay_test, delay_test_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_delay_test, delay_test_pred, zero_division=0
        ),
        "F1 score": f1_score(
            y_delay_test, delay_test_pred, zero_division=0
        ),
        "Delay rate": y_delay_test.mean(),
        "Average planned journey (hours)": (
            test_df["planned_transit_days"].mean() * 24
        ),
    },
])

col1, col2, col3 = st.columns(3)

col1.metric(
    "Training examples",
    f"{len(train_df):,}",
    help="Examples used to fit the models.",
)

col2.metric(
    "Testing examples",
    f"{len(test_df):,}",
    help="Held-out examples used to evaluate the models.",
)

col3.metric(
    "Delay definition",
    "Over 24 hours",
    help=(
        "A shipment is delayed when its simulated final delay "
        "exceeds 24 hours."
    ),
)

tab_data, tab_models = st.tabs([
    "Compare the data",
    "Compare model performance",
])


# ------------------- DATA COMPARISON ---------------------

with tab_data:

    st.caption(
        "Check whether the training and testing groups have similar "
        "delay rates and planned journey durations."
    )

    rate_df = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "Delayed shipments (%)": [
            100 * performance.loc[0, "Delay rate"],
            100 * performance.loc[1, "Delay rate"],
        ],
    })

    rate_fig = px.line(
        rate_df,
        x="Dataset",
        y="Delayed shipments (%)",
        markers=True,
        text="Delayed shipments (%)",
        title="Share of shipments delayed by more than 24 hours",
        labels={
            "Dataset": "",
            "Delayed shipments (%)": "Delayed shipments (%)",
        },
        template=plotly_template,
    )

    rate_fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="top center",
        line={"width": 3},
        marker={"size": 12},
    )

    rate_fig.update_layout(
        showlegend=False,
        yaxis={"rangemode": "tozero"},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    st.plotly_chart(rate_fig, use_container_width=True)

    duration_df = pd.concat([
        train_df[["planned_transit_days"]].assign(
            Dataset="Training"
        ),
        test_df[["planned_transit_days"]].assign(
            Dataset="Testing"
        ),
    ], ignore_index=True)

    # Convert durations to hours for the chart.
    duration_df["planned_transit_hours"] = (
        duration_df["planned_transit_days"] * 24
    )

    duration_fig = px.box(
        duration_df,
        x="Dataset",
        y="planned_transit_hours",
        points="outliers",
        title="Spread of planned journey durations",
        labels={
            "Dataset": "",
            "planned_transit_hours": "Planned journey duration (hours)",
        },
        template=plotly_template,
    )

    duration_fig.update_layout(
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    st.plotly_chart(duration_fig, use_container_width=True)

    st.caption(
        "The box shows the middle half of journey durations. The line "
        "inside is the median; points outside the whiskers may be "
        "unusually short or long."
    )


# ---------------- MODEL PERFORMANCE COMPARISON ------------

with tab_models:

    st.caption(
        "Lower ETA error is better. For classification scores, values "
        "closer to 1 are generally better."
    )

    eta_compare = performance[[
        "Dataset",
        "ETA MAE (hours)",
    ]].copy()

    eta_fig = px.line(
        eta_compare,
        x="Dataset",
        y="ETA MAE (hours)",
        markers=True,
        text="ETA MAE (hours)",
        title="ETA prediction error (MAE)",
        labels={
            "Dataset": "",
            "ETA MAE (hours)": "Average absolute error (hours)",
        },
        template=plotly_template,
    )

    eta_fig.update_traces(
        texttemplate="%{text:.1f} hours",
        textposition="top center",
        line={"width": 3},
        marker={"size": 12},
    )

    eta_fig.update_layout(
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    st.plotly_chart(eta_fig, use_container_width=True)

    classification_compare = performance.melt(
        id_vars="Dataset",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 score",
        ],
        var_name="Metric",
        value_name="Score",
    )

    classification_fig = px.line(
        classification_compare,
        x="Dataset",
        y="Score",
        color="Metric",
        markers=True,
        text="Score",
        title="Delay prediction scores",
        labels={
            "Dataset": "",
            "Score": "Score (0 to 1)",
            "Metric": "Measure",
        },
        template=plotly_template,
    )

    classification_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="top center",
        marker={"size": 8},
    )

    classification_fig.update_layout(
        yaxis={"range": [0, 1.08]},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    st.plotly_chart(
        classification_fig,
        use_container_width=True,
    )

    display_metrics = performance[[
        "Dataset",
        "ETA MAE (hours)",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
    ]].copy()

    display_metrics["ETA MAE (hours)"] = (
        display_metrics["ETA MAE (hours)"]
        .map(lambda value: f"{value:.1f} hours")
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
    ]:
        display_metrics[column] = (
            display_metrics[column]
            .map(lambda value: f"{value:.1%}")
        )

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "If training scores are much better than testing scores, the model "
    "may be overfitting. These results use one split of synthetic data "
    "and do not guarantee real-world performance."
)


# =========================================================
# 12. SHIPMENT DETAILS
# =========================================================

with st.expander("View shipment details and units"):

    st.caption(
        "Journey durations are displayed in hours. The trained model "
        "and source dataset retain their original day-based units internally."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Journey information**")

        st.write(f"- Shipment ID: {shipment['shipment_id']}")
        st.write(f"- Starting point: {shipment['origin']}")
        st.write(f"- Destination: {shipment['destination']}")

        st.write(
            f"- Planned journey: "
            f"{shipment['planned_transit_days'] * 24:.1f} hours"
        )

        st.write(
            f"- Typical journey: "
            f"{shipment['base_transit_days'] * 24:.1f} hours"
        )

        st.write(
            f"- Delay before departure: "
            f"{shipment['departure_delay_hours']:.1f} hours"
        )

    with right:

        st.markdown("**Condition inputs**")

        st.write(
            f"- Weather severity: {shipment['weather_severity']} "
            "(simulated level)"
        )

        st.write(
            f"- Congestion: {shipment['congestion_level']} "
            "(simulated level)"
        )

        st.write(
            f"- Customs risk: {shipment['customs_risk']} "
            "(simulated setting)"
        )

        st.write(
            f"- Route sensitivity to weather: "
            f"{shipment['weather_sensitivity']} "
            "(model input score; no physical unit)"
        )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "ShipMind AI · Synthetic-data prototype · For demonstration and "
    "learning, not live shipment decisions."
)