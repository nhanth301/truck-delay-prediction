import json
from typing import Any, Dict

import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfRowsWithMissingValues,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
)
from workflow.schema import State


def normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all datetime columns in a DataFrame are converted to datetime64[ns] without timezone.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized datetime columns.
    """
    # Handle timezone-aware datetime columns
    datetime_cols = df.select_dtypes(
        include=["datetimetz", "datetime64[ns, UTC]", "datetime64[us, UTC]"]
    ).columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

    # Try parsing object columns as datetime
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.notna().any():
                df[col] = parsed.dt.tz_localize(None)
        except Exception:
            # Ignore columns that cannot be parsed as datetime
            pass

    return df


def run_data_quality_check(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run data quality tests comparing reference and current datasets.

    Args:
        reference_data (pd.DataFrame): Reference dataset.
        current_data (pd.DataFrame): Current dataset.

    Returns:
        Dict[str, Any]: JSON-style dictionary with test results.
    """
    reference_data = normalize_datetime(reference_data.copy())
    current_data = normalize_datetime(current_data.copy())

    tests = TestSuite(
        tests=[
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
        ]
    )
    tests.run(reference_data=reference_data, current_data=current_data)
    return json.loads(tests.json())


def assert_quality_passed(suite: Dict[str, Any], logger: Any) -> bool:
    """
    Check if all quality tests passed.

    Args:
        suite (Dict[str, Any]): Test results.
        logger (Any): Logger from state.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    failed_tests = [t for t in suite["tests"] if t["status"] == "FAIL"]

    if failed_tests:
        names = [t["name"] for t in failed_tests]
        logger.error("❌ Data Quality Fail: %s", names)
        return False

    logger.info("✅ Data Quality Passed")
    return True


def check_data_quality(state: State) -> Dict[str, Any]:
    """
    Check data quality for new data against reference feature groups.

    Args:
        state (State): State containing feature group data, new data, and logger.

    Returns:
        Dict[str, Any]: Results of data quality checks and routing decision.
    """
    logger = state["logger"]
    fgs_data = state["feature_groups_data"]

    fg_to_key = {
        "truck_schedule_details_fg": "truck_schedule",
        "traffic_details_fg": "traffic",
        "route_weather_details_fg": "route_weather",
        "city_weather_details_fg": "city_weather",
    }

    new_data_quality: Dict[str, bool] = {}

    for fg, ref_df in fgs_data.items():
        if fg not in fg_to_key:
            continue

        key = fg_to_key[fg]
        logger.info("Running data quality check for: %s", key)

        suite = run_data_quality_check(ref_df, state["new_data"][key])
        new_data_quality[key] = assert_quality_passed(suite, logger)

    should_continue = any(new_data_quality.values())

    return {
        "new_data_quality": new_data_quality,
        "should_continue": should_continue,
    }


def data_quality_router(state: State) -> str:
    """
    Route execution flow depending on data quality check results.

    Args:
        state (State): State containing 'should_continue'.

    Returns:
        str: 'proceed' if quality checks passed, 'terminate' otherwise.
    """
    return "proceed" if state["should_continue"] else "terminate"
