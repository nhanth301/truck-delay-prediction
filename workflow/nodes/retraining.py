from typing import Dict, Any
from pipeline.utils import split_data_by_date, calculate_class_weights
from pipeline.process import processing_data
from pipeline.modeling import train_models
from workflow.schema import State


def train(state: State) -> Dict[str, Any]:
    """
    Train models using the final dataset.

    Steps:
        1. Split dataset into train/validation/test.
        2. Process datasets (scaling, encoding, etc.).
        3. Calculate class weights for imbalance handling.
        4. Train models and log results.

    Args:
        state (State): Workflow state containing config and final_data.

    Returns:
        Dict[str, Any]: Training status and optionally evaluation metrics.
    """
    logger = state["logger"]

    try:
        config = state["config"]
        final_df = state["final_data"]

        # 1. Split dataset
        train_df, val_df, test_df = split_data_by_date(final_df, config)
        logger.info(
            "✅ Data split complete: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df)
        )

        # 2. Preprocess data
        X_train, y_train, X_valid, y_valid, X_test, y_test = processing_data(
            config, train_df, val_df, test_df
        )
        logger.info("✅ Data preprocessing complete.")

        # 3. Compute class weights
        class_weights = calculate_class_weights(y_train)
        logger.info("✅ Calculated class weights: %s", class_weights)

        # 4. Train models
        train_models(
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            class_weights,
            config
        )
        logger.info("🎯 Training completed successfully.")

        return {"train_status": "success"}

    except Exception as e:
        logger.exception("❌ Training failed with error: %s", str(e))
        return {"train_status": "failed", "error": str(e)}
