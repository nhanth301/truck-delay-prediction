from pipeline.utils import fetch_data
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.tests import *
import json
from workflow.schema import State
import hopsworks
import pandas as pd 
def fetch_fs(config,fs):
    fgs = [
        "truck_schedule_details_fg",
        "traffic_details_fg",
        "route_weather_details_fg",
        "city_weather_details_fg",
    ]
    return {fg: fetch_data(config,fs,fg) for fg in fgs}

def normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all datetime columns are datetime64[ns] without timezone"""
    for col in df.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]", "datetime64[us, UTC]"]).columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.notna().any():
                df[col] = parsed.dt.tz_localize(None)
        except Exception:
            pass
    return df

def run_data_quality_check(reference_data, current_data):
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

def assert_quality_passed(suite):
    failed_tests = [t for t in suite["tests"] if t["status"] == "FAIL"]
    if failed_tests:
        names = [t["name"] for t in failed_tests]
        print(f"❌ Data Quality Fail: {names}")
        return False
    print("✅ Data Quality Passed")
    return True

def check_data_quality(state: State):
    project = hopsworks.login(api_key_value=state['config']['hopsworks']['api_key'])
    fs = project.get_feature_store()
    fgs_data = fetch_fs(state['config'],fs)

    fg_to_key = {
        "truck_schedule_details_fg": "truck_schedule",
        "traffic_details_fg": "traffic",
        "route_weather_details_fg": "route_weather",
        "city_weather_details_fg": "city_weather",
    }

    new_data_quality = {}
    for fg, ref_df in fgs_data.items():
        if fg not in fg_to_key:
            raise ValueError(f"Unexpected feature group: {fg}")
        key = fg_to_key[fg]
        print('Key data quality check:',key)
        suite = run_data_quality_check(ref_df, state['new_data'][key])
        new_data_quality[key] = assert_quality_passed(suite)

    should_continue = any(new_data_quality.values())

    return {"new_data_quality": new_data_quality,
            'should_continue': should_continue}

def data_quality_router(state: State):
    if state['should_continue']:
        return 'continue'
    return 'END'

    


