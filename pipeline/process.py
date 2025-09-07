import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pickle import dump
import logging
from pipeline.utils import setup_logging, drop_null_values, fill_missing_values_with_mode, process_categorical_data, scale_data

logger = setup_logging()


def save_encoder_scaler(encoder, scaler, encoder_path='files/truck_data_encoder.pkl', scaler_path='files/truck_data_scaler.pkl'):
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

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

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

        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        encoder.fit(train[encode_columns])

        def safe_process(df):
            encoded_array = encoder.transform(df[encode_columns])
            if not isinstance(encoded_array, np.ndarray):
                encoded_array = encoded_array.toarray()
            encoded_cols = encoder.get_feature_names_out(encode_columns)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=encode_columns), encoded_df], axis=1)
            return df

        train = safe_process(train)
        valid = safe_process(valid)
        test = safe_process(test)

        scaler = StandardScaler()
        train[cts_cols] = scaler.fit_transform(train[cts_cols])
        valid = scale_data(valid, scaler, cts_cols)
        test = scale_data(test, scaler, cts_cols)

        save_encoder_scaler(encoder, scaler)

        onehot_cols = list(encoder.get_feature_names_out(encode_columns))
        keep_cat_cols = [c for c in cat_cols if c not in encode_columns]  # giữ cat chưa encode
        feature_cols = cts_cols + keep_cat_cols + onehot_cols

        X_train, y_train = train[feature_cols], train[target]
        X_valid, y_valid = valid[feature_cols], valid[target]
        X_test, y_test = test[feature_cols], test[target]

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    except Exception as e:
        logger.error(f"An error occurred while processing the data: {str(e)}")
        return None, None, None, None, None, None