from typing import Any, Dict

from workflow.schema import State
from pipeline.data_prep import processing_and_feature_engg


def update_fs(config: Dict[str, Any], fs: Any, feature_group_name: str, new_data: Any, logger: Any) -> None:
    """
    Insert new data into a given feature group.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        fs (Any): Feature store instance.
        feature_group_name (str): Name of the feature group.
        new_data (Any): DataFrame containing new data.
        logger (Any): Logger from state.
    """
    feature_group = fs.get_feature_group(
        feature_group_name,
        version=config["hopsworks"]["feature_group_version"],
    )
    feature_group.insert(new_data)
    logger.info("✅ Inserted %d rows into %s", len(new_data), feature_group_name)


def rebuild_merged_feature_group(fgs_data: Dict[str, Any], config: Dict[str, Any], fs: Any, logger: Any) -> None:
    """
    Rebuild the merged feature group from multiple feature groups.

    Args:
        fgs_data (Dict[str, Any]): Dictionary of feature group DataFrames.
        config (Dict[str, Any]): Configuration dictionary.
        fs (Any): Feature store instance.
        logger (Any): Logger from state.
    """
    routes_df = fgs_data["routes_details_fg"]
    trucks_df = fgs_data["truck_details_fg"]
    drivers_df = fgs_data["drivers_details_fg"]
    truck_schedule_df = fgs_data["truck_schedule_details_fg"]
    traffic_df = fgs_data["traffic_details_fg"]
    city_weather_df = fgs_data["city_weather_details_fg"]
    route_weather_df = fgs_data["route_weather_details_fg"]

    fs_data = processing_and_feature_engg(
        routes_df,
        route_weather_df,
        drivers_df,
        trucks_df,
        traffic_df,
        truck_schedule_df,
        city_weather_df,
        0,
    )

    logger.info("🔄 Rebuilding merged feature group with %d rows...", len(fs_data))

    if fs_data.empty:
        logger.warning("⚠️ No data to insert into final_data")
    else:
        update_fs(config, fs, "final_data", fs_data, logger)


def update_feature_store(state: State) -> Dict[str, Any]:
    """
    Update the feature store with new data and rebuild the merged feature group.

    Args:
        state (State): State containing configuration, feature store, and new data.

    Returns:
        Dict[str, Any]: Update status for each feature group.
    """
    logger = state["logger"]
    fs = state["feature_store"]

    key_to_fg = {
        "truck_schedule": "truck_schedule_details_fg",
        "traffic": "traffic_details_fg",
        "route_weather": "route_weather_details_fg",
        "city_weather": "city_weather_details_fg",
    }

    update_status: Dict[str, str] = {}

    for key, fg_name in key_to_fg.items():
        if state["new_data_quality"][key] and state["new_data_status"][key]:
            update_fs(state["config"], fs, fg_name, state["new_data"][key], logger)
            update_status[key] = f"Inserted {len(state['new_data'][key])} rows"
        else:
            update_status[key] = "Skipped (quality fail or no data)"
            logger.info("⏭️ Skipped update for %s (quality fail or no new data)", key)

    rebuild_merged_feature_group(state["feature_groups_data"], state["config"], fs, logger)
    update_status["final_data"] = "Rebuilt"

    return {"update_status": update_status}


def is_the_first_day_of_week(state: State) -> str:
    """
    Router function to decide if today is the first day of the week.

    Args:
        state (State): State containing constants.

    Returns:
        str: 'proceed' if it is the first day of the week, otherwise 'terminate'.
    """
    return "proceed" if state["constant"]["dayofweek"] == 1 else "terminate"
