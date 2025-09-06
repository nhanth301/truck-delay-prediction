import pandas as pd
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, DataDriftTable
from evidently import ColumnMapping
from pipeline.utils import fetch_data


def check_data_drift(state):
    """
    Node to check data drift between last 7 days and the rest of the data in final_data.
    """

    fs = state['feature_store']
    final_df = fetch_data(state['config'], fs, 'final_data')
    if final_df is None or final_df.empty:
        print("⚠️final_data is empty, skipping drift check.")
        return {"data_drift": {"drifted": False, "details": None}}

    final_df["estimated_arrival"] = pd.to_datetime(final_df["estimated_arrival"])

    cutoff_date = final_df["estimated_arrival"].max() - timedelta(days=7)

    reference_data = final_df[final_df["estimated_arrival"] < cutoff_date].copy()
    current_data   = final_df[final_df["estimated_arrival"] >= cutoff_date].copy()

    if reference_data.empty or current_data.empty:
        print("⚠️ Not enough data to compare drift.")
        return {"data_drift": {"drifted": False, "details": None}}

    target = "Delay"
    prediction = "Delay_Predictions"

    numerical_features = [
        'route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip',
        'route_avg_humidity', 'route_avg_visibility', 'route_avg_pressure',
        'distance', 'average_hours', 'origin_temp', 'origin_wind_speed',
        'origin_precip', 'origin_humidity', 'origin_pressure',
        'destination_temp','destination_wind_speed','destination_precip',
        'destination_humidity','destination_pressure','avg_no_of_vehicles',
        'truck_age','load_capacity_pounds','mileage_mpg',
        'age','experience','average_speed_mph'
    ]
    categorical_features = [
        'origin_visibility','destination_visibility',
        'accident','ratings','is_midnight'
    ]

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    report = Report(metrics=[
        DatasetDriftMetric(drift_share=0.1),
        DataDriftTable()
    ])

    report.run(
        reference_data=reference_data.drop(columns=["unique_id"], errors="ignore"),
        current_data=current_data.drop(columns=["unique_id"], errors="ignore"),
        column_mapping=column_mapping
    )

    drift_results = report.as_dict()
    drifted = drift_results["metrics"][0]["result"]["dataset_drift"]

    if drifted:
        print("❌ Data drift detected between last 7 days and historical data.")
    else:
        print("✅ No significant data drift detected.")

    return {
        "data_drift": {
            "drifted": drifted,
            "details": drift_results
        }
    }
