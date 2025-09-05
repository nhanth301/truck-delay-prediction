from pipeline.data_prep import create_postgres_connection, read_data, create_mysql_connection
from workflow.schema import State
import pandas as pd

def init_db_conn(config):
    postgres_conn = create_postgres_connection(config)
    mysql_conn = create_mysql_connection(config)
    return postgres_conn, mysql_conn

def ensure_datetime(df, col):
    """Convert column to pandas datetime64[ns] safely, drop tz"""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    return df

def filter(df, date, is_route_weather=False, is_schedule=False):
    ts = pd.to_datetime(date).tz_localize(None)

    if is_route_weather:
        df = ensure_datetime(df, 'Date')
        return df[df['Date'] > ts]
    elif is_schedule:
        df = ensure_datetime(df, 'departure_date')
        return df[df['departure_date'] > ts]
    else:
        df = ensure_datetime(df, 'date')
        return df[df['date'] > ts]

def check_new_data(state: State):
    postgres_conn, mysql_conn = init_db_conn(state['config'])
    tracking_status = read_data(postgres_conn, '"CONSTANT"').iloc[0].to_dict()
    mysql_tables = ['traffic_details', 'truck_schedule_data', 'city_weather']
    postgres_tables = ['routes_weather']

    traffic_df = read_data(mysql_conn, mysql_tables[0])
    truck_schedule_df = read_data(mysql_conn, mysql_tables[1])
    city_weather_df = read_data(mysql_conn, mysql_tables[2])
    route_weather_df = read_data(postgres_conn, postgres_tables[0])

    new_traffic_df = filter(traffic_df, tracking_status['date'])
    new_truck_schedule_df = filter(truck_schedule_df, tracking_status['date'], is_schedule=True)
    new_city_weather_df = filter(city_weather_df, tracking_status['date'])
    new_route_weather_df = filter(route_weather_df, tracking_status['date'], is_route_weather=True)

    status = {
        'traffic': len(new_traffic_df) > 0,
        'truck_schedule': len(new_truck_schedule_df) > 0,
        'city_weather': len(new_city_weather_df) > 0,
        'route_weather': len(new_route_weather_df) > 0
    }

    print(f'traffic: {len(new_traffic_df)} rows added!!')
    print(f'truck_schedule: {len(new_truck_schedule_df)} rows added!!')
    print(f'city_weather: {len(new_city_weather_df)} rows added!!')
    print(f'route_weather: {len(new_route_weather_df)} rows added!!')

    if all(len(df) == 0 for df in [new_traffic_df, new_truck_schedule_df, new_city_weather_df, new_route_weather_df]):
        return {'should_continue': False}
    else:
        return {
            'new_data': {
                'traffic': new_traffic_df,
                'truck_schedule': new_truck_schedule_df,
                'city_weather': new_city_weather_df,
                'route_weather': new_route_weather_df
            },
            'should_continue': True,
            'new_data_status': status
        }

def new_data_router(state: State):
    return 'continue' if state['should_continue'] else 'END'
