from pipeline.utils import *
from pipeline.modeling import *
from pipeline.process import processing_data
import os
import sys
import warnings
warnings.filterwarnings('ignore')


def main():
    config = load_config()
    logger = setup_logging()
    try:
        final_merge = fetch_data(config, 'final_data')
        logger.info("Final merged data fetched from Hopsworks.")
        
        train_df, validation_df, test_df = split_data_by_date(final_merge, config)
        logger.info("Data split into train test and validation sets.")
        

        X_train, y_train, X_valid, y_valid, X_test, y_test = processing_data(config, train_df, validation_df, test_df)

        class_weights = calculate_class_weights(y_train)

        os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']

        logger.info("Model Training started.")
        train_models(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, config)
        logger.info("Model Training and Registration Completed")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit()

if __name__ == '__main__':
    main()