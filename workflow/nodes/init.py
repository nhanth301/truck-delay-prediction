from pipeline.data_prep import create_postgres_connection, read_data, create_mysql_connection
import hopsworks
from pipeline.utils import fetch_data
from workflow.schema import State
def init_db_conn(config):
    postgres_conn = create_postgres_connection(config)
    mysql_conn = create_mysql_connection(config)
    return postgres_conn, mysql_conn

def init_fs(config):
    project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
    fs = project.get_feature_store()
    return fs

def fetch_fs(config, fs):
    fgs = [
        "truck_schedule_details_fg",
        "traffic_details_fg",
        "route_weather_details_fg",
        "city_weather_details_fg",
        'drivers_details_fg',
        'routes_details_fg',
        'truck_details_fg'
    ]
    return {fg: fetch_data(config,fs,fg) for fg in fgs}

def init_node(state: State):
    postgres_conn, mysql_conn = init_db_conn(state['config'])
    feature_store = init_fs(state['config'])
    fgs = fetch_fs(state['config'],feature_store)
    return {
        'db_conn': {
            'pg': postgres_conn,
            'mysql': mysql_conn
        },
        'feature_store': feature_store,
        'feature_groups_data': fgs
    }

