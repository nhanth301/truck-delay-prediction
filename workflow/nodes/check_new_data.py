from pipeline.data_prep import create_postgres_connection, read_data, create_mysql_connection
from workflow.schema import State


def init_db_conn(config):
    postgres_conn = create_postgres_connection(config)
    mysql_conn = create_mysql_connection(config)
    return postgres_conn, mysql_conn

def filter(df,date,is_route_weather=False):
    if is_route_weather:
        return df[df['Date'] > date]
    else:
        return df[df['date'] > date]

def check_new_data(state: State):
    postgres_conn, mysql_conn = init_db_conn(state['config'])
    tracking_status = read_data(postgres_conn, '"CONSTANT"').iloc[0].to_dict()
    mysql_tables = ['traffic_details', 'truck_schedule_data', 'city_weather']
    postgres_tables = ['routes_weather']

    traffic_df = read_data(mysql_conn,mysql_tables[0])
    truck_schedule_df = read_data(mysql_conn,mysql_tables[1])
    city_weather_df = read_data(mysql_conn,mysql_tables[2])
    route_weather_df = read_data(postgres_conn,postgres_tables[0])

    new_traffic_df = filter(traffic_df,tracking_status['date'])
    new_truck_schedule_df = filter(truck_schedule_df,tracking_status['date'])
    new_city_weather_df = filter(city_weather_df,tracking_status['date'])
    new_route_weather_df = filter(route_weather_df,tracking_status['date'],True)

    print(f'traffic: {len(new_traffic_df)} rows added!!')
    print(f'truck_schedule: {len(new_truck_schedule_df)} rows added!!')
    print(f'city_weather: {len(new_city_weather_df)} rows added!!')
    print(f'route_weather: {len(new_route_weather_df)} rows added!!')

    if len(new_traffic_df) and len(new_truck_schedule_df) and len(new_city_weather_df) and len(new_route_weather_df):
        return {'continue': False}
    else:
        return {'new_data': {'traffic' : new_traffic_df, 
                             'truck_schedule': new_truck_schedule_df, 
                             'city_weather': new_city_weather_df, 
                             'route_weather': new_route_weather_df}}


    


