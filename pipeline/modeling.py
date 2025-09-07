import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import logging
from pipeline.utils import setup_logging

import joblib
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

logger = setup_logging()

def train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, SWEEP_CONFIG, PROJECT_NAME):
    '''
    Train a Logistic Regression model and log performance metrics using Weights & Biases.
    '''
    try:
        logger.info("Preparing sweep configuration...")
        configs = {
            "method": SWEEP_CONFIG['method'],
            "metric": {
                "name": SWEEP_CONFIG['metric']['name'],
                "goal": SWEEP_CONFIG['metric']['goal']
            },
            "parameters": {
                "penalty": {"values": SWEEP_CONFIG['parameters']['penalty']['values']},
                "C": {"values": SWEEP_CONFIG['parameters']['C']['values']}
            }
        }

        logger.info("Initializing W&B sweep...")
        sweep_id = wandb.sweep(sweep=configs, project=PROJECT_NAME)
        logger.info(f"Sweep created with ID: {sweep_id}")

        def train():
            with wandb.init(project=PROJECT_NAME) as run:
                try:
                    config = wandb.config
                    logger.info(f"Starting training with config: {dict(config)}")

                    model = LogisticRegression(
                        penalty=config['penalty'],
                        C=config['C'],
                        class_weight=class_weights,
                        solver='liblinear'
                    )

                    logger.info("Fitting Logistic Regression model...")
                    model.fit(X_train, y_train)

                    logger.info("Making predictions...")
                    y_train_pred = model.predict(X_train)
                    y_valid_pred = model.predict(X_valid)
                    y_test_pred = model.predict(X_test)
                    y_test_proba = model.predict_proba(X_test)

                    scores = {
                        "f1_score_train": f1_score(y_train, y_train_pred),
                        "f1_score_valid": f1_score(y_valid, y_valid_pred),
                        "f1_score": f1_score(y_test, y_test_pred),
                    }
                    logger.info(f"Scores: {scores}")
                    wandb.log(scores)

                    wandb.sklearn.plot_classifier(
                        model, X_train, X_test, y_train, y_test,
                        y_test_pred, y_test_proba,
                        labels=None, model_name='LogisticRegression',
                        feature_names=X_train.columns
                    )

                    logger.info("Saving trained model...")
                    model_artifact = wandb.Artifact(
                        "LogisticRegression", type="model", metadata=dict(config)
                    )
                    joblib.dump(model, "models/log-truck-model.pkl")
                    model_artifact.add_file("models/log-truck-model.pkl")
                    run.log_artifact(model_artifact)

                    logger.info("Training run completed successfully.")

                except Exception as e:
                    logger.error(f"Error in training run: {e}", exc_info=True)
                    wandb.log({"error": str(e)})
                    raise e  # re-raise to stop sweep agent

        wandb.agent(sweep_id=sweep_id, function=train, project=PROJECT_NAME)

    except Exception as e:
        logger.error(f"Error in sweep setup: {e}", exc_info=True)
        wandb.log({"error": str(e)})

def train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, SWEEP_CONFIG, PROJECT_NAME):
    '''
    Train a Random Forest classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    try:
        logger.info("Preparing sweep configuration...")
        configs = {
            "method": SWEEP_CONFIG['method'],
            "metric": {
                "name": SWEEP_CONFIG['metric']['name'],
                "goal": SWEEP_CONFIG['metric']['goal']
            },
            "parameters": {
                "n_estimators": {"values": SWEEP_CONFIG['parameters']['n_estimators']['values']},
                "max_depth": {"values": SWEEP_CONFIG['parameters']['max_depth']['values']},
                "min_samples_split": {"values": SWEEP_CONFIG['parameters']['min_samples_split']['values']}
            }
        }

        logger.info("Initializing W&B sweep...")
        sweep_id = wandb.sweep(sweep=configs, project=PROJECT_NAME)
        logger.info(f"Sweep created with ID: {sweep_id}")

        def train():
            with wandb.init(project=PROJECT_NAME) as run:
                try:
                    config = wandb.config
                    logger.info(f"Starting training with config: {dict(config)}")

                    model = RandomForestClassifier(
                        n_estimators=config['n_estimators'],
                        max_depth=config['max_depth'],
                        min_samples_split=config['min_samples_split'],
                        class_weight=class_weights
                    )

                    logger.info("Fitting Random Forest model...")
                    model.fit(X_train, y_train)

                    logger.info("Making predictions...")
                    y_train_pred = model.predict(X_train)
                    y_valid_pred = model.predict(X_valid)
                    y_test_pred = model.predict(X_test)
                    y_test_proba = model.predict_proba(X_test)

                    scores = {
                        "f1_score_train": f1_score(y_train, y_train_pred),
                        "f1_score_valid": f1_score(y_valid, y_valid_pred),
                        "f1_score": f1_score(y_test, y_test_pred),
                    }
                    logger.info(f"Scores: {scores}")
                    wandb.log(scores)

                    wandb.sklearn.plot_classifier(
                        model, X_train, X_test, y_train, y_test,
                        y_test_pred, y_test_proba,
                        labels=None, model_name='RandomForest',
                        feature_names=X_train.columns
                    )

                    logger.info("Saving trained model...")
                    model_artifact = wandb.Artifact(
                        "RandomForest", type="model", metadata=dict(config)
                    )
                    joblib.dump(model, "models/randf-truck-model.pkl")
                    model_artifact.add_file("models/randf-truck-model.pkl")
                    run.log_artifact(model_artifact)

                    logger.info("Training run completed successfully.")

                except Exception as e:
                    logger.error(f"Error in training run: {e}", exc_info=True)
                    wandb.log({"error": str(e)})
                    raise e  # re-raise to stop sweep agent

        wandb.agent(sweep_id=sweep_id, function=train, project=PROJECT_NAME)
    except Exception as e:
        logger.error(f"Error in sweep setup: {e}", exc_info=True)
        wandb.log({"error": str(e)})


def train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, SWEEP_CONFIG, PROJECT_NAME):
    '''
    Train an XGBoost classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    try:
        logger.info("Preparing sweep configuration...")
        configs = {
            "method": SWEEP_CONFIG['method'],
            "metric": {
                "name": SWEEP_CONFIG['metric']['name'],
                "goal": SWEEP_CONFIG['metric']['goal']
            },
            "parameters": {
                "learning_rate": {"values": SWEEP_CONFIG['parameters']['learning_rate']['values']},
                "max_depth_xgb": {"values": SWEEP_CONFIG['parameters']['max_depth_xgb']['values']},
                "n_estimators_xgb": {"values": SWEEP_CONFIG['parameters']['n_estimators_xgb']['values']}
            }
        }

        logger.info("Initializing W&B sweep...")
        sweep_id = wandb.sweep(sweep=configs, project=PROJECT_NAME)
        logger.info(f"Sweep created with ID: {sweep_id}")
        def train():
            with wandb.init(project=PROJECT_NAME) as run:
                try:
                    config = wandb.config
                    logger.info(f"Starting training with config: {dict(config)}")

                    xgbmodel = XGBClassifier(
                        objective="binary:logistic",
                        learning_rate=config.learning_rate,
                        max_depth=config.max_depth_xgb,
                        n_estimators=config.n_estimators_xgb,
                        random_state=42 
                    )
                    xgbmodel.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False             
                    )

                    logger.info("Making predictions...")
                    y_train_pred = xgbmodel.predict(X_train)
                    y_valid_pred = xgbmodel.predict(X_valid)
                    y_test_pred = xgbmodel.predict(X_test)
                    y_test_proba = xgbmodel.predict_proba(X_test)

                    scores = {
                        "f1_score_train": f1_score(y_train, y_train_pred, average="weighted"),
                        "f1_score_valid": f1_score(y_valid, y_valid_pred, average="weighted"),
                        "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
                    }
                    logger.info(f"Scores: {scores}")
                    wandb.log(scores)
                    wandb.sklearn.plot_classifier(
                        xgbmodel, X_train, X_test, y_train, y_test,
                        y_test_pred, y_test_proba,
                        labels=None, model_name='XGBoost',
                        feature_names=X_train.columns
                    )

                    logger.info("Saving trained model...")
                    model_artifact = wandb.Artifact(
                        "XGBoost", type="model", metadata=dict(config)
                    )
                    joblib.dump(xgbmodel, "models/xgb-truck-model.pkl")
                    model_artifact.add_file("models/xgb-truck-model.pkl")
                    run.log_artifact(model_artifact)

                    logger.info("Training run completed successfully.")

                except Exception as e:
                    logger.error(f"Error in training run: {e}", exc_info=True)
                    wandb.log({"error": str(e)})
                    raise e 

        wandb.agent(sweep_id=sweep_id, function=train, project=PROJECT_NAME)
    except Exception as e:
        logger.error(f"Error in sweep setup: {e}", exc_info=True)
        wandb.log({"error": str(e)})
    

def train_models(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train multiple machine learning models and log performance metrics.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.
    - project_config (dict): Configuration parameters for the project.

    Returns:
    - None
    '''
    try:
        PROJECT_NAME = project_config['wandb']['wandb_project']
        # Train Logistic Regression model
        train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config['sweep'], PROJECT_NAME)

        # Train Random Forest model
        train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config['sweep'], PROJECT_NAME)

        # Train XGBoost model
        train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config['sweep'], PROJECT_NAME)

    except Exception as e:
        logger.error(f"An error occurred while training the models: {str(e)}")