from workflow.schema import State
import hopsworks

def init_fs(config):
    project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
    fs = project.get_feature_store()
    return fs 

def update_fs(config, fs, feature_group_name, new_data):
    feature_group = fs.get_feature_group(
        feature_group_name,
        version=config['hopsworks']['feature_group_version']
    )
    feature_group.insert(new_data)
    print(f"✅ Inserted {len(new_data)} rows into {feature_group_name}")

def update_feature_store(state: State):
    fs = init_fs(state['config'])
    key_to_fg = {
        "truck_schedule": "truck_schedule_details_fg",
        "traffic": "traffic_details_fg",
        "route_weather": "route_weather_details_fg",
        "city_weather":"city_weather_details_fg"
    }

    for key in key_to_fg:
        if state['new_data_quality'][key] and state['new_data_status'][key]:
            update_fs(state['config'], fs, key_to_fg[key], state["new_data"][key])
