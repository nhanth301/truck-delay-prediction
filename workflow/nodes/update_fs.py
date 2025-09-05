from workflow.schema import State
from pipeline.data_prep import processing_and_feature_engg


def update_fs(config, fs, feature_group_name, new_data):
    feature_group = fs.get_feature_group(
        feature_group_name,
        version=config['hopsworks']['feature_group_version']
    )
    feature_group.insert(new_data)
    print(f"✅ Inserted {len(new_data)} rows into {feature_group_name}")

def rebuild_merged_feature_group(fgs_data, config, fs):
    routes_df = fgs_data['routes_details_fg']
    trucks_df = fgs_data['truck_details_fg']
    drivers_df = fgs_data['drivers_details_fg']
    truck_schedule_df = fgs_data['truck_schedule_details_fg']
    traffic_df = fgs_data['traffic_details_fg']
    city_weather_df = fgs_data['city_weather_details_fg']
    route_weather_df = fgs_data['route_weather_details_fg']
    fs_data = processing_and_feature_engg(routes_df,route_weather_df,drivers_df,trucks_df,traffic_df,truck_schedule_df,city_weather_df,0)
    print(f"🔄 Rebuilding merged feature group with {len(fs_data)} rows...")
    if fs_data.empty:
        print("⚠️ No data to insert into final_data")
    else:
        update_fs(config, fs, 'final_data', fs_data)

def update_feature_store(state: State):
    fs = state['feature_store']
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

    rebuild_merged_feature_group(state['feature_groups_data'], state['config'], fs)
    update_status['final_data'] = "Rebuilt"

    return {"update_status": update_status}

def is_the_first_day(state: State):
    if state['constant']['dayofweek'] == 1:
        return 'continue'
    else:
        return 'END'