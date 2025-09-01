import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from sklearn.metrics import f1_score
import logging
from pipeline.utils import setup_logging

logger = setup_logging()

def train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train a Logistic Regression model and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    features = X_train.columns
    PROJECT_NAME = project_config['wandb']['wandb_project']
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        params = {
            "random_state": 13,
            "class_weight": class_weights 
            }

        model = LogisticRegression(**params)

        model.fit(X_train, y_train)

        # Train predictions and performance
        y_train_pred = model.predict(X_train)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = model.predict(X_valid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})

        # Plot classifier performance
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test,
                                            y_preds, y_probas, labels= None, model_name='LogisticRegression', feature_names=features)

        # Save the trained model
        model_artifact = wandb.Artifact("LogisticRegression", type="model", metadata=dict(config))
        joblib.dump(model, "models/log-truck-model.pkl")
        model_artifact.add_file("models/log-truck-model.pkl")
        wandb.save("models/log-truck-model.pkl")
        run.log_artifact(model_artifact)


