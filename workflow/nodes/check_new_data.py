from typing import Any, Dict

import pandas as pd

from pipeline.data_prep import read_data
from workflow.schema import State


def ensure_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert a column in a DataFrame to pandas datetime64[ns] safely, removing timezone.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to convert.

    Returns:
        pd.DataFrame: DataFrame with the specified column converted to datetime.
    """
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.tz_localize(None)
    return df


def filter_new_data(
    df: pd.DataFrame,
    date: str,
    is_route_weather: bool = False,
    is_schedule: bool = False,
) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows with dates later than the given timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date (str): Reference date string.
        is_route_weather (bool, optional): If True, use 'Date' column.
        is_schedule (bool, optional): If True, use 'departure_date' column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    timestamp = pd.to_datetime(date).tz_localize(None)

    if is_route_weather:
        df = ensure_datetime(df, "Date")
        return df[df["Date"] > timestamp]

    if is_schedule:
        df = ensure_datetime(df, "departure_date")
        return df[df["departure_date"] > timestamp]

    df = ensure_datetime(df, "date")
    return df[df["date"] > timestamp]


def check_new_data(state: State) -> Dict[str, Any]:
    """
    Check for new data across multiple tables since the last tracking date.

    Args:
        state (State): State containing database connections, logger, and other context.

    Returns:
        Dict[str, Any]: Status and optionally new data if available.
    """
    logger = state["logger"]
    postgres_conn = state["db_conn"]["postgres"]
    mysql_conn = state["db_conn"]["mysql"]

    tracking_status = read_data(postgres_conn, '"CONSTANT"').iloc[0].to_dict()

    mysql_tables = ["traffic_details", "truck_schedule_data", "city_weather"]
    postgres_tables = ["routes_weather"]

    traffic_df = read_data(mysql_conn, mysql_tables[0])
    truck_schedule_df = read_data(mysql_conn, mysql_tables[1])
    city_weather_df = read_data(mysql_conn, mysql_tables[2])
    route_weather_df = read_data(postgres_conn, postgres_tables[0])

    new_traffic_df = filter_new_data(traffic_df, tracking_status["date"])
    new_truck_schedule_df = filter_new_data(
        truck_schedule_df, tracking_status["date"], is_schedule=True
    )
    new_city_weather_df = filter_new_data(city_weather_df, tracking_status["date"])
    new_route_weather_df = filter_new_data(
        route_weather_df, tracking_status["date"], is_route_weather=True
    )

    status = {
        "traffic": len(new_traffic_df) > 0,
        "truck_schedule": len(new_truck_schedule_df) > 0,
        "city_weather": len(new_city_weather_df) > 0,
        "route_weather": len(new_route_weather_df) > 0,
    }

    logger.info("Traffic: %d new rows detected", len(new_traffic_df))
    logger.info("Truck schedule: %d new rows detected", len(new_truck_schedule_df))
    logger.info("City weather: %d new rows detected", len(new_city_weather_df))
    logger.info("Route weather: %d new rows detected", len(new_route_weather_df))

    if all(len(df) == 0 for df in [
        new_traffic_df,
        new_truck_schedule_df,
        new_city_weather_df,
        new_route_weather_df,
    ]):
        return {"should_continue": False}

    return {
        "new_data": {
            "traffic": new_traffic_df,
            "truck_schedule": new_truck_schedule_df,
            "city_weather": new_city_weather_df,
            "route_weather": new_route_weather_df,
        },
        "should_continue": True,
        "new_data_status": status,
        "constant": tracking_status,
    }


def new_data_router(state: State) -> str:
    """
    Route execution flow depending on whether new data is available.

    Args:
        state (State): State dictionary containing 'should_continue'.

    Returns:
        str: 'proceed' if new data is found, 'terminate' otherwise.
    """
    return "proceed" if state["should_continue"] else "terminate"
