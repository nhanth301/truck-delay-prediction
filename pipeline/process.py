import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pickle import dump
import logging
from pipeline.utils import setup_logging, drop_null_values, fill_missing_values_with_mode, process_categorical_data, scale_data

logger = setup_logging()


def save_encoder_scaler(encoder, scaler, encoder_path='output/truck_data_encoder.pkl', scaler_path='output/truck_data_scaler.pkl'):
    try:
        dump(encoder, open(encoder_path, 'wb'))
        dump(scaler, open(scaler_path, 'wb'))
        logger.info("Encoder and scaler saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during saving encoder and scaler: {str(e)}")

def calculate_mode(data, column_name):
    try:
        mode_value = data[column_name].mode().iloc[0]
        logger.info(f"Mode calculated successfully for {column_name}.")

        return mode_value

    except Exception as e:
        logger.error(f"An error occurred during mode calculation: {str(e)}")

def processing_data(config, train, valid, test):

    try:
        columns_to_drop_nulls = config['features']['columns_to_drop_null_values']
        encode_columns = config['features']['encode_column_names']
        cts_cols = config['features']['cts_col_names']
        cat_cols = config['features']['cat_col_names']
        target = config['features']['target']

        train = drop_null_values(train, columns_to_drop_nulls)
        valid = drop_null_values(valid, columns_to_drop_nulls)
        test = drop_null_values(test, columns_to_drop_nulls)

        load_capacity_mode = calculate_mode(train, 'load_capacity_pounds')


        fill_missing_values_with_mode(train, 'load_capacity_pounds', load_capacity_mode)
        fill_missing_values_with_mode(valid, 'load_capacity_pounds', load_capacity_mode)
        fill_missing_values_with_mode(test, 'load_capacity_pounds', load_capacity_mode)


        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        encoder.fit(train[encode_columns])
        train = process_categorical_data(train, encoder, encode_columns)
        valid = process_categorical_data(valid, encoder, encode_columns)
        test = process_categorical_data(test, encoder, encode_columns)

        scaler = StandardScaler()
        train[cts_cols] = scaler.fit_transform(train[cts_cols])
        valid = scale_data(valid, scaler, cts_cols)
        test = scale_data(test, scaler, cts_cols)

        save_encoder_scaler(encoder, scaler)

        X_train = train[cts_cols + cat_cols]
        y_train = train[target]

        X_valid = valid[cts_cols + cat_cols]
        y_valid = valid[target]

        X_test = test[cts_cols + cat_cols]
        y_test = test[target]

        X_train = X_train.drop(encode_columns, axis=1)
        X_valid = X_valid.drop(encode_columns, axis=1)
        X_test = X_test.drop(encode_columns, axis=1)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    except Exception as e:
        logger.error(f"An error occurred while processing the data: {str(e)}")