import pandas as pd
from datetime import timedelta
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, DataDriftTable
from evidently import ColumnMapping
from pipeline.utils import fetch_data
from workflow.schema import State
from typing import Any, Dict


def check_data_drift(state: State) -> Dict[str, Any]:
    """
    Node to check data drift between the last 7 days and historical data in `final_data`.

    Args:
        state (State): Current workflow state containing config and feature store.

    Returns:
        Dict[str, Any]: A dictionary with data drift status and details.
    """
    logger = state["logger"]
    fs = state["feature_store"]

    # Fetch merged feature group data
    final_df = fetch_data(state["config"], fs, "final_data")
    if final_df is None or final_df.empty:
        logger.warning("⚠️ final_data is empty, skipping drift check.")
        return {"data_drift": {"drifted": False, "details": None}}

    # Ensure datetime
    final_df["estimated_arrival"] = pd.to_datetime(final_df["estimated_arrival"])

    # Split into reference and current datasets
    cutoff_date = final_df["estimated_arrival"].max() - timedelta(days=7)
    reference_data = final_df[final_df["estimated_arrival"] < cutoff_date].copy()
    current_data = final_df[final_df["estimated_arrival"] >= cutoff_date].copy()

    if reference_data.empty or current_data.empty:
        logger.warning("⚠️ Not enough data to perform drift check.")
        return {"data_drift": {"drifted": False, "details": None}}

    # Define schema
    target = "Delay"
    prediction = "Delay_Predictions"

    numerical_features = [
        "route_avg_temp", "route_avg_wind_speed", "route_avg_precip",
        "route_avg_humidity", "route_avg_visibility", "route_avg_pressure",
        "distance", "average_hours", "origin_temp", "origin_wind_speed",
        "origin_precip", "origin_humidity", "origin_pressure",
        "destination_temp", "destination_wind_speed", "destination_precip",
        "destination_humidity", "destination_pressure", "avg_no_of_vehicles",
        "truck_age", "load_capacity_pounds", "mileage_mpg",
        "age", "experience", "average_speed_mph",
    ]
    categorical_features = [
        "origin_visibility", "destination_visibility",
        "accident", "ratings", "is_midnight",
    ]

    column_mapping = ColumnMapping(
        target=target,
        prediction=prediction,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Build Evidently report
    report = Report(metrics=[
        DatasetDriftMetric(drift_share=0.1),
        DataDriftTable()
    ])

    report.run(
        reference_data=reference_data.drop(columns=["unique_id"], errors="ignore"),
        current_data=current_data.drop(columns=["unique_id"], errors="ignore"),
        column_mapping=column_mapping,
    )

    drift_results = report.as_dict()
    drifted = drift_results["metrics"][0]["result"]["dataset_drift"]

    if drifted:
        logger.error("❌ Data drift detected between last 7 days and historical data.")
    else:
        logger.info("✅ No significant data drift detected.")

    return {
        "data_drift": {
            "drifted": drifted,
            "details": drift_results,
        },
        "final_data": final_df,
    }


def data_drift_router(state: State) -> str:
    """
    Router node to decide next step based on drift results.

    Args:
        state (State): Current workflow state.

    Returns:
        str: 'trigger_retrain' if drift detected, otherwise 'proceed'.
    """
    return "trigger_retrain" if state["data_drifted"]["drifted"] else "proceed"
