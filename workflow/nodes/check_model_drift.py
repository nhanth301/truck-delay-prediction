import os
import pickle
import joblib
import pandas as pd
from sklearn.metrics import f1_score
from typing import Any, Dict, Tuple
from pipeline.utils import processing_new_data, fetch_best_model, fetch_data
from workflow.schema import State


def load_model(file_path: str) -> Any:
    """
    Load a pickled object from disk.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: The loaded Python object.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_pkl_from_artifact(artifact_dir: str) -> str:
    """
    Get the path of the first `.pkl` file in a given directory.

    Args:
        artifact_dir (str): Path to the artifact directory.

    Returns:
        str: Full path to the `.pkl` file.

    Raises:
        FileNotFoundError: If no `.pkl` file is found.
    """
    pkl_files = sorted([f for f in os.listdir(artifact_dir) if f.endswith(".pkl")])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {artifact_dir}")
    return os.path.join(artifact_dir, pkl_files[0])


def init_model(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """
    Initialize ML model, scaler, and encoder from artifacts.

    Args:
        config (Dict[str, Any]): Workflow configuration.

    Returns:
        Tuple[Any, Any, Any]: (model, scaler, encoder)
    """
    model_artifact = fetch_best_model(config)
    model_path = get_pkl_from_artifact(model_artifact)
    model = joblib.load(model_path)

    scaler = load_model(config["preprocess_model"]["scaler"])
    encoder = load_model(config["preprocess_model"]["encoder"])

    return model, scaler, encoder


def check_model_drift(state: State) -> Dict[str, Any]:
    """
    Check model drift by comparing F1 scores between historical reference data
    and the most recent 7 days.

    Args:
        state (State): Workflow state containing config and final_data.

    Returns:
        Dict[str, Any]: Drift results including F1 scores and relative drop.
    """
    logger = state["logger"]
    config = state["config"]
    final_data = state["final_data"]

    model, scaler, encoder = init_model(config)

    if final_data is None or final_data.empty:
        logger.warning("⚠️ final_data is empty, skipping model drift check.")
        return {"model_drift": {"drifted": False, "details": None}}

    final_data["estimated_arrival"] = pd.to_datetime(final_data["estimated_arrival"])
    cutoff_date = final_data["estimated_arrival"].max() - pd.Timedelta(days=7)

    reference_df = final_data[final_data["estimated_arrival"] < cutoff_date].copy()
    current_df = final_data[final_data["estimated_arrival"] >= cutoff_date].copy()

    if reference_df.empty or current_df.empty:
        logger.warning("⚠️ Not enough data to compare model drift.")
        return {"model_drift": {"drifted": False, "details": None}}

    # Compute F1 scores
    X_ref, y_ref = processing_new_data(config, reference_df, scaler, encoder)
    f1_ref = f1_score(y_ref, model.predict(X_ref))

    X_cur, y_cur = processing_new_data(config, current_df, scaler, encoder)
    f1_cur = f1_score(y_cur, model.predict(X_cur))

    relative_drop = (f1_ref - f1_cur) / max(f1_ref, 1e-6)
    drop_threshold = config.get("drift", {}).get("relative_drop_threshold", 0.1)
    drifted = relative_drop > drop_threshold

    if drifted:
        logger.error(
            "❌ Model drift detected: F1 dropped %.1f%% (ref=%.3f → cur=%.3f)",
            relative_drop * 100, f1_ref, f1_cur
        )
    else:
        logger.info(
            "✅ No model drift: F1 stable (ref=%.3f → cur=%.3f)", f1_ref, f1_cur
        )

    return {
        "model_drift": {
            "drifted": drifted,
            "f1_ref": f1_ref,
            "f1_cur": f1_cur,
            "relative_drop": relative_drop,
            "threshold": drop_threshold,
            "sizes": {
                "reference": len(reference_df),
                "current": len(current_df),
            },
        }
    }


def model_drift_router(state: State) -> str:
    """
    Router node to decide next step based on model drift results.

    Args:
        state (State): Workflow state.

    Returns:
        str: 'trigger_retrain' if drift detected, otherwise 'terminate'.
    """
    return "trigger_retrain" if state["model_drift"]["drifted"] else "terminate"
