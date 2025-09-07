import logging
from typing import Any, Dict, Tuple

import hopsworks

from pipeline.data_prep import (
    create_postgres_connection,
    create_mysql_connection,
)
from pipeline.utils import fetch_data, setup_logging
from workflow.schema import State


def init_logger() -> logging.Logger:
    """
    Initialize and return a logger instance.

    Returns:
        logging.Logger: Configured logger object.
    """
    return setup_logging()


def init_db_conn(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Initialize database connections for Postgres and MySQL.

    Args:
        config (Dict[str, Any]): Configuration dictionary with database credentials.

    Returns:
        Tuple[Any, Any]: Postgres and MySQL connection objects.
    """
    postgres_conn = create_postgres_connection(config)
    mysql_conn = create_mysql_connection(config)
    return postgres_conn, mysql_conn


def init_feature_store(config: Dict[str, Any]) -> Any:
    """
    Initialize the Hopsworks feature store.

    Args:
        config (Dict[str, Any]): Configuration dictionary with Hopsworks API key.

    Returns:
        Any: Hopsworks feature store object.
    """
    project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
    return project.get_feature_store()


def fetch_feature_groups(config: Dict[str, Any], feature_store: Any) -> Dict[str, Any]:
    """
    Fetch feature groups data from the feature store.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        feature_store (Any): Hopsworks feature store object.

    Returns:
        Dict[str, Any]: Dictionary mapping feature group names to their data.
    """
    feature_groups = [
        "truck_schedule_details_fg",
        "traffic_details_fg",
        "route_weather_details_fg",
        "city_weather_details_fg",
        "drivers_details_fg",
        "routes_details_fg",
        "truck_details_fg",
    ]
    return {fg: fetch_data(config, feature_store, fg) for fg in feature_groups}


def init_node(state: State) -> Dict[str, Any]:
    """
    Initialize system components: logger, database connections, feature store, and feature groups.

    Args:
        state (State): Application state containing configuration.

    Returns:
        Dict[str, Any]: Initialized components including database connections,
                        feature store, feature groups data, and logger.
    """
    logger = init_logger()
    postgres_conn, mysql_conn = init_db_conn(state["config"])
    feature_store = init_feature_store(state["config"])
    feature_groups_data = fetch_feature_groups(state["config"], feature_store)

    return {
        "db_conn": {
            "postgres": postgres_conn,
            "mysql": mysql_conn,
        },
        "feature_store": feature_store,
        "feature_groups_data": feature_groups_data,
        "logger": logger,
    }
