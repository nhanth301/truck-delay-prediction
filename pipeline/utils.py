import logging
import yaml
import hopsworks
import pandas as pd
import os
import wandb
import numpy as np
from dotenv import load_dotenv


def setup_logging():
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s')
    
    error_log_path = 'logs/truck_eta_error_logs.log' 
    file_handler_error=logging.FileHandler(os.path.abspath(error_log_path))
    file_handler_error.setFormatter(formatter)
    file_handler_error.setLevel(logging.ERROR)
    logger.addHandler(file_handler_error)
    
    info_log_path = 'logs/truck_eta_info_logs.log' 
    file_handler_info=logging.FileHandler(os.path.abspath(info_log_path))
    file_handler_info.setFormatter(formatter)
    file_handler_info.setLevel(logging.INFO)
    logger.addHandler(file_handler_info)
    
    return logger

# logger = setup_logging()

def load_config(file_path='config.yaml'):
    """
    Load configuration from a YAML file.
    
    Parameters:
    file_path (str): The path to the YAML file. Default is 'config.yaml'.
    
    Returns:
    dict: The configuration as a dictionary.
    """
    load_dotenv()
    with open(file_path, 'r') as config_file:
        raw_content = config_file.read()
        expanded_content = os.path.expandvars(raw_content)
        config = yaml.safe_load(expanded_content)
        
    return config

def fetch_data(config, feature_group_name):
    """
    Fetch data from a feature group in Hopsworks.
    
    Parameters:
    config (dict): The configuration as a dictionary.
    
    Returns:
    pandas.DataFrame: The data as a pandas DataFrame.
    """

    project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
    fs = project.get_feature_store()

    try:
        feature_group = fs.get_feature_group(
            feature_group_name,
            version=config['hopsworks']['feature_group_version']
        )

        query = feature_group.select_all()
        data = query.read(read_options={"use_hive": config['hopsworks']['use_hive_option']})
        return data

    except Exception as e:
        logger.error(f"An error occurred during data fetching: {str(e)}")


def split_data_by_date(final_merge, config):
    """
    This function splits the data into training, validation, and test sets based on the given date ranges.
    
    Parameters:
    final_merge (DataFrame): The merged DataFrame.
    config (dict): The configuration dictionary.
    
    Returns:
    tuple: A tuple containing six DataFrames (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    try:
        # Specify date ranges
        train_end_date = config['split_date_ranges']['train_end_date']
        test_start_date = config['split_date_ranges']['test_start_date']
    

        # Splitting the data into training, validation, and test sets based on date
        train_df = final_merge[final_merge['estimated_arrival'] <= pd.to_datetime(train_end_date)]
        validation_df = final_merge[
            (final_merge['estimated_arrival'] > pd.to_datetime(train_end_date)) &
            (final_merge['estimated_arrival'] <= pd.to_datetime(test_start_date))
        ]
        test_df = final_merge[final_merge['estimated_arrival'] > pd.to_datetime(test_start_date)]


        return train_df, validation_df, test_df

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during data splitting: {str(e)}")


def calculate_class_weights(y_train):
    '''
    Calculate class weights based on the distribution of target labels.
    
    Parameters:
    - y_train: Target labels for training data.
    
    Returns:
    - class_weights (dict): Dictionary containing class weights.
    '''
    class_counts = y_train.value_counts().to_dict()
    
    weights = len(y_train) / (2 * class_counts[0]), len(y_train) / (2 * class_counts[1])
    class_weights = {0: weights[0], 1: weights[1]}
    
    return class_weights


def fetch_best_model(config):
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']
    PROJECT_NAME = config['wandb']['wandb_project']
    USER_NAME= config['wandb']['wandb_user']
    run = wandb.init(project=PROJECT_NAME)

    runs = wandb.Api().runs(f"{run.entity}/{PROJECT_NAME}")

    best_f1_score = -1
    best_run_id = None

    print(f"Run scores comparison starting")
    for run in runs:
        run_metrics = run.history()

        if "f1_score_train" in run_metrics.columns and "f1_score_valid" in run_metrics.columns and "f1_score" in run_metrics.columns:
            f1_score_train = run_metrics['f1_score_train'].max()
            f1_score_valid = run_metrics['f1_score_valid'].max()
            f1_score_test = run_metrics['f1_score'].max()

            avg_f1_score = np.median([f1_score_train , f1_score_valid ,f1_score_test])

            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                best_run_id = run.id


    # Print the best F1 score and its corresponding run ID
    print(f"Best Average F1 Score: {best_f1_score}")
    print(f"Best Run ID: {best_run_id}")

    # Fetch the details of the best run
    best_run = wandb.Api().run(f"{run.entity}/{PROJECT_NAME}/{best_run_id}")

    artifacts = best_run.logged_artifacts()
    best_model = [artifact for artifact in artifacts if artifact.type == 'model'][0]

    artifact_dir = best_model.download()

    logger.info("artifact_dir: %s", artifact_dir)

    return artifact_dir


def drop_null_values(data, columns_to_drop):
    try:
        processed_data = data.dropna(subset=columns_to_drop).reset_index(drop=True)
        print("Null values dropped successfully.")

        return processed_data

    except Exception as e:
        print(f"An error occurred during null value removal: {str(e)}")

def fill_missing_values_with_mode(data, column_name, mode_value):
    try:
        data[column_name] = data[column_name].fillna(mode_value)
        print(f"Missing values in {column_name} filled with mode value.")

    except Exception as e:
        print(f"An error occurred during missing value imputation: {str(e)}")


def process_categorical_data(data, encoder, encode_columns):
    try:
        encoded_features = list(encoder.get_feature_names_out(encode_columns))
        data[encoded_features] = encoder.transform(data[encode_columns])
        print("One-Hot Encoding completed successfully.")


        print("Original categorical features dropped successfully.")

        return data

    except Exception as e:
        print(f"An error occurred during categorical data processing: {str(e)}")

def scale_data(data, scaler, cts_cols):
    try:
        data[cts_cols] = scaler.transform(data[cts_cols])
        print("Data scaling completed successfully.")

        return data

    except Exception as e:
        print(f"An error occurred during data scaling: {str(e)}")

def processing_new_data(config, data, scaler, encoder):

    try:
        columns_to_drop_nulls = config['features']['columns_to_drop_null_values']
        encode_columns = config['features']['encode_column_names']
        cts_cols = config['features']['cts_col_names']
        cat_cols = config['features']['cat_col_names']
        target = config['features']['target']

        data = drop_null_values(data, columns_to_drop_nulls)

        fill_missing_values_with_mode(data, 'load_capacity_pounds', 3000)

        data = process_categorical_data(data, encoder, encode_columns)

        data = scale_data(data, scaler, cts_cols)

        X_test = data[cts_cols + cat_cols]
        y_test = data[target]

        X_test = X_test.drop(encode_columns, axis=1)

        return X_test, y_test

    except Exception as e:
        print(f"An error occurred while processing the data: {str(e)}")