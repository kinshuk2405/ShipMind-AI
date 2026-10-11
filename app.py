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

from src.explainability import (
    explain_delay_risk,
    explain_delay_risk_shap,
    explain_delay_risk_lime,
    get_global_shap_importance,
)

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


st.sidebar.header("2. Scenario planner")

st.sidebar.caption(
    "Choose a hypothetical intervention to compare its modeled impact."
)

scenario_presets = {
    "Current plan": (0, 0),
    "Improve sea transit": (20, 0),
    "Improve destination processing": (0, 20),
    "Combined intervention": (20, 20),
    "Custom scenario": None,
}

scenario_choice = st.sidebar.selectbox(
    "Select an intervention",
    list(scenario_presets.keys()),
)

if scenario_choice == "Custom scenario":
    sea_reduction = st.sidebar.slider(
        "Reduce sea-stage excess time",
        min_value=0,
        max_value=80,
        value=20,
        step=5,
        format="%d%%",
    )

    destination_reduction = st.sidebar.slider(
        "Reduce destination-stage excess time",
        min_value=0,
        max_value=80,
        value=20,
        step=5,
        format="%d%%",
    )
else:
    sea_reduction, destination_reduction = (
        scenario_presets[scenario_choice]
    )

st.sidebar.caption(
    f"Assumptions: sea transit −{sea_reduction}%, "
    f"destination processing −{destination_reduction}%."
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


baseline_trace = pd.DataFrame(baseline["trace"])
scenario_trace = pd.DataFrame(scenario["trace"])

stage_comparison = baseline_trace[
    ["stage", "baseline_excess_hours"]
].merge(
    scenario_trace[["stage", "scenario_excess_hours"]],
    on="stage",
)

stage_names = {
    "Origin handling": "Origin handling",
    "Sea transit": "Sea transit",
    "Destination processing": "Destination processing",
    "Final handling": "Final handling",
}

stage_comparison["Stage"] = (
    stage_comparison["stage"]
    .map(stage_names)
    .fillna(stage_comparison["stage"])
)

stage_comparison = stage_comparison.rename(
    columns={
        "baseline_excess_hours": "Before intervention (hours)",
        "scenario_excess_hours": "After intervention (hours)",
    }
)

chart_data = stage_comparison.melt(
    id_vars="Stage",
    value_vars=[
        "Before intervention (hours)",
        "After intervention (hours)",
    ],
    var_name="Scenario",
    value_name="Extra time (hours)",
)

stage_fig = px.bar(
    chart_data,
    x="Stage",
    y="Extra time (hours)",
    color="Scenario",
    barmode="group",
    title="Stage-level impact of the selected scenario",
    labels={"Stage": "", "Extra time (hours)": "Excess time (hours)"},
    template=plotly_template,
)

stage_fig.update_layout(
    margin={"l": 10, "r": 10, "t": 45, "b": 10},
    legend_title_text="",
)

st.plotly_chart(stage_fig, use_container_width=True)


col1, col2 = st.columns(2)

col1.metric(
    "Schedule allowance",
    f"{baseline['schedule_buffer_hours']:.1f} hours",
)

col2.metric(
    "Remaining delay after intervention",
    f"{scenario['delay_after_buffer_hours']:.1f} hours",
)


# =========================================================
# 9. WHAT-IF SCENARIO
# =========================================================

st.divider()
st.markdown("## 3. Scenario impact")

st.write(
    "Compare the modeled journey duration before and after the "
    "selected hypothetical intervention."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Baseline modeled ETA",
    f"{baseline['baseline_modeled_eta_hours']:.1f} hours",
)

col2.metric(
    "Scenario modeled ETA",
    f"{scenario['scenario_eta_hours']:.1f} hours",
)

col3.metric(
    "Modeled time saved",
    f"{scenario['hours_saved_vs_original']:.1f} hours",
)

col1, col2 = st.columns(2)

col1.metric(
    "Baseline delay after allowance",
    f"{baseline['baseline_delay_after_buffer_hours']:.1f} hours",
)

col2.metric(
    "Scenario delay after allowance",
    f"{scenario['delay_after_buffer_hours']:.1f} hours",
)

st.info(
    f"Selected scenario: {scenario_choice}. "
    f"Assumed reduction in sea-stage excess time: {sea_reduction}%. "
    f"Assumed reduction in destination-stage excess time: "
    f"{destination_reduction}%."
)

st.caption(
    "These are simulated results based on assumed reductions in "
    "stage-level excess durations. The scenario ETA is not a new "
    "prediction from the trained ETA model, and modeled savings "
    "are not guaranteed real-world outcomes."
)



# =========================================================
# AUTOMATIC INTERVENTION RANKING
# =========================================================

st.markdown("### Which intervention has the greatest modeled impact?")

st.write(
    "ShipMind AI compares a consistent set of hypothetical "
    "interventions for the selected shipment."
)

candidate_scenarios = [
    {
        "Intervention": "Current plan",
        "Sea reduction (%)": 0,
        "Destination reduction (%)": 0,
    },
    {
        "Intervention": "Improve sea transit",
        "Sea reduction (%)": 20,
        "Destination reduction (%)": 0,
    },
    {
        "Intervention": "Improve destination processing",
        "Sea reduction (%)": 0,
        "Destination reduction (%)": 20,
    },
    {
        "Intervention": "Combined intervention",
        "Sea reduction (%)": 20,
        "Destination reduction (%)": 20,
    },
]

ranking_rows = []

for candidate in candidate_scenarios:
    result = propagate_delay(
        shipment,
        sea_improvement=candidate["Sea reduction (%)"] / 100,
        destination_improvement=(
            candidate["Destination reduction (%)"] / 100
        ),
    )

    ranking_rows.append({
        "Intervention": candidate["Intervention"],
        "Sea reduction (%)": candidate["Sea reduction (%)"],
        "Destination reduction (%)": (
            candidate["Destination reduction (%)"]
        ),
        "Modeled ETA (hours)": result["scenario_eta_hours"],
        "Remaining delay (hours)": result["delay_after_buffer_hours"],
        "Modeled hours saved": result["hours_saved_vs_original"],
    })

ranking_df = pd.DataFrame(ranking_rows)

# Rank the hypothetical options by modeled time saved.
ranking_df = ranking_df.sort_values(
    "Modeled hours saved",
    ascending=False,
).reset_index(drop=True)

ranking_df.insert(
    0,
    "Rank",
    range(1, len(ranking_df) + 1),
)

st.dataframe(
    ranking_df.style.format({
        "Sea reduction (%)": "{:.0f}%",
        "Destination reduction (%)": "{:.0f}%",
        "Modeled ETA (hours)": "{:.2f}",
        "Remaining delay (hours)": "{:.2f}",
        "Modeled hours saved": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Highlight the highest-ranked scenario.
best = ranking_df.iloc[0]

if best["Intervention"] != "Current plan":
    st.success(
        f"Highest modeled impact: {best['Intervention']} — "
        f"{best['Modeled hours saved']:.2f} hours saved "
        "under the stated assumptions."
    )
else:
    st.info(
        "No intervention produces modeled time savings for this "
        "shipment under the current assumptions."
    )

ranking_fig = px.bar(
    ranking_df.sort_values("Modeled hours saved", ascending=True),
    x="Modeled hours saved",
    y="Intervention",
    orientation="h",
    text="Modeled hours saved",
    title="Modeled time saved by intervention",
    labels={
        "Modeled hours saved": "Hours saved",
        "Intervention": "",
    },
    template=plotly_template,
)

ranking_fig.update_traces(
    texttemplate="%{text:.2f} h",
    textposition="outside",
)

ranking_fig.update_layout(
    showlegend=False,
    margin={"l": 10, "r": 55, "t": 45, "b": 10},
)

st.plotly_chart(
    ranking_fig,
    use_container_width=True,
)

st.caption(
    "Ranking is based on assumed reductions in excess stage durations, "
    "not learned intervention effects. It identifies the largest "
    "modeled time saving, not necessarily the cheapest or most feasible "
    "real-world action."
)


# =========================================================
# 10. MODEL EXPLAINABILITY — SHAP AND LIME
# =========================================================

st.divider()
st.markdown("## 4. Why did the model flag this shipment?")
st.write(
    "Explore which inputs influenced the delay-risk classifier. "
    "SHAP explains contributions relative to a background baseline; "
    "LIME provides a local approximation around the selected shipment."
)

st.info(
    "These explanations describe the model's behavior on synthetic data. "
    "They do not establish real-world causes of shipment delays."
)


def friendly_name(name):
    label = (
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
    )

    replacements = {
        "departure delay hours": "Departure delay",
        "weather severity": "Weather severity",
        "congestion level": "Congestion level",
        "customs risk": "Customs risk",
        "weather sensitivity": "Weather sensitivity",
        "planned transit days": "Planned transit duration",
        "base transit days": "Typical transit duration",
        "route id": "Route",
        "origin": "Origin",
        "destination": "Destination",
    }

    for raw, friendly in replacements.items():
        label = label.replace(raw, friendly)

    return label


tab_global, tab_shap, tab_lime = st.tabs(
    ["Global SHAP", "Selected shipment: SHAP", "Selected shipment: LIME"]
)

# ---------------- GLOBAL SHAP ----------------

with tab_global:
    st.markdown("### Which features influence predictions most?")
    st.caption(
        "Mean absolute SHAP values across a reproducible sample. "
        "Larger values indicate greater average contribution magnitude."
    )

    try:
        global_importance = get_global_shap_importance(
            sample_size=200
        ).head(12).copy()

        global_importance["Feature"] = (
            global_importance["feature"].map(friendly_name)
        )

        global_fig = px.bar(
            global_importance.sort_values("mean_absolute_shap"),
            x="mean_absolute_shap",
            y="Feature",
            orientation="h",
            title="Global feature importance using SHAP",
            labels={
                "mean_absolute_shap": "Mean absolute SHAP value",
                "Feature": "",
            },
            template=plotly_template,
        )

        global_fig.update_layout(
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
        )

        st.plotly_chart(global_fig, use_container_width=True)

        st.caption(
            "Global importance measures contribution magnitude, not "
            "whether a feature generally increases or decreases risk."
        )

    except Exception as exc:
        st.error(f"Could not calculate global SHAP importance: {exc}")


# ---------------- LOCAL SHAP ----------------

with tab_shap:
    st.markdown("### What pushed this shipment's risk estimate?")
    st.caption(
        f"Shipment: {selected_id}. Positive SHAP values push the "
        "model output toward higher delay risk; negative values push it lower."
    )

    try:
        shap_result = explain_delay_risk_shap(shipment).head(12).copy()
        shap_result["Feature"] = shap_result["feature"].map(friendly_name)
        shap_result["Direction"] = shap_result["shap_value"].apply(
            lambda value: (
                "Higher modeled risk" if value > 0
                else "Lower modeled risk" if value < 0
                else "Neutral"
            )
        )

        shap_fig = px.bar(
            shap_result.sort_values("shap_value"),
            x="shap_value",
            y="Feature",
            color="Direction",
            orientation="h",
            title="Local SHAP contributions",
            labels={
                "shap_value": "SHAP value (model output scale)",
                "Feature": "",
                "Direction": "",
            },
            template=plotly_template,
        )

        shap_fig.update_layout(
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
        )

        st.plotly_chart(shap_fig, use_container_width=True)
        st.dataframe(
            shap_result[["Feature", "shap_value", "Direction"]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Could not calculate the SHAP explanation: {exc}")


# ---------------- LOCAL LIME ----------------

with tab_lime:
    st.markdown("### A second local explanation")
    st.caption(
        "LIME approximates the classifier around this shipment. "
        "Its weights are not directly comparable to SHAP values."
    )

    try:
        lime_result = explain_delay_risk_lime(
            shipment,
            num_features=10,
        ).copy()

        lime_result["Feature"] = lime_result["feature"].map(friendly_name)

        lime_fig = px.bar(
            lime_result.sort_values("lime_weight"),
            x="lime_weight",
            y="Feature",
            color="direction",
            orientation="h",
            title="LIME local feature weights",
            labels={
                "lime_weight": "LIME weight (local approximation)",
                "Feature": "",
                "direction": "",
            },
            template=plotly_template,
        )

        lime_fig.update_layout(
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
        )

        st.plotly_chart(lime_fig, use_container_width=True)

        st.dataframe(
            lime_result[["Feature", "lime_weight", "direction"]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Could not calculate the LIME explanation: {exc}")


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