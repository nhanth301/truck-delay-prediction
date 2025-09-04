from pipeline.utils import fetch_data
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.tests import *
import json
from workflow.schema import State

def fetch_fs(config):
    fgs = [
        "truck_schedule_details_fg",
        "traffic_details_fg",
        "route_weather_details_fg",
        "city_weather_details_fg",
    ]
    return {fg: fetch_data(config, fg) for fg in fgs}

def run_data_quality_check(reference_data, current_data):
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
    fgs_data = fetch_fs(state['config'])

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
        suite = run_data_quality_check(ref_df, state['new_data'][key])
        new_data_quality[key] = assert_quality_passed(suite)

    return {"new_data_quality": new_data_quality}
    


