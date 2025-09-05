from workflow.schema import State
import hopsworks
from pipeline.utils import fetch_data
from pipeline.data_prep import processing_and_feature_engg
def init_fs(config):
    project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
    fs = project.get_feature_store()
    return fs 

def update_fs(config, fs, feature_group_name, new_data, overwrite=False):
    feature_group = fs.get_feature_group(
        feature_group_name,
        version=config['hopsworks']['feature_group_version']
    )
    feature_group.insert(new_data,overwrite=overwrite)
    print(f"✅ Inserted {len(new_data)} rows into {feature_group_name}")

def rebuild_merged_feature_group(config, fs):
    routes_df = fetch_data(config, fs, 'routes_details_fg')
    trucks_df = fetch_data(config, fs, 'truck_details_fg')
    drivers_df = fetch_data(config, fs, 'drivers_details_fg')
    truck_schedule_df = fetch_data(config, fs, 'truck_schedule_details_fg')
    traffic_df = fetch_data(config, fs, 'traffic_details_fg')
    city_weather_df = fetch_data(config, fs, 'city_weather_details_fg')
    route_weather_df = fetch_data(config, fs, 'route_weather_details_fg')
    fs_data = processing_and_feature_engg(routes_df,route_weather_df,drivers_df,trucks_df,traffic_df,truck_schedule_df,city_weather_df,0)
    print(f"🔄 Rebuilding merged feature group with {len(fs_data)} rows...")
    if fs_data.empty:
        print("⚠️ No data to insert into final_data")
    else:
        update_fs(config, fs, 'final_data', fs_data, overwrite=False)

def update_feature_store(state: State):
    fs = init_fs(state['config'])
    key_to_fg = {
        "truck_schedule": "truck_schedule_details_fg",
        "traffic": "traffic_details_fg",
        "route_weather": "route_weather_details_fg",
        "city_weather":"city_weather_details_fg"
    }

    update_status = {}
    for key in key_to_fg:
        if state['new_data_quality'][key] and state['new_data_status'][key]:
            update_fs(state['config'], fs, key_to_fg[key], state["new_data"][key])
            update_status[key] = f"Inserted {len(state['new_data'][key])} rows"
        else:
            update_status[key] = "Skipped (quality fail or no data)"

    rebuild_merged_feature_group(state['config'], fs)
    update_status['final_data'] = "Rebuilt"

    return {"update_status": update_status}
