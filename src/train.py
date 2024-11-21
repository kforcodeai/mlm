import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass
import json
from .models import ModelRegistry
import pandas
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path

mlflow.set_tracking_uri("http://0.0.0.0:8080")

@dataclass
class ModelMetrics:
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float
    train_roc_auc: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_roc_auc: float
    
@dataclass
class ModelResult:
    model_name: str
    model: Any
    metrics: ModelMetrics
    model_size: int
    model_path: Path

def get_model_size(model) -> int:
    """Calculate approximate model size in bytes"""
    import pickle
    return len(pickle.dumps(model))

def evaluate_model(model, X, y) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    """
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = None
    
    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1': f1_score(y, y_pred, average='weighted'),
    }
    
    # Add ROC AUC if applicable (binary classification and predict_proba available)
    if y_pred_proba is not None and len(np.unique(y)) == 2:
        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
    else:
        metrics['roc_auc'] = 0.0
    
    # Get confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def log_metrics_to_mlflow(metrics: Dict[str, float], m_path: str = "", prefix: str = ""):
    """
    Log metrics to MLflow with optional prefix
    """
    for metric_name, value in metrics.items():
        if metric_name != 'confusion_matrix':
            mlflow.log_metric(f"{prefix}{metric_name}", value)
        else:
            # Log confusion matrix as a JSON artifact
            cm_path = str(m_path / f"confusion_matrix_{prefix.strip('_')}.json")
            with open(cm_path, 'w') as f:
                json.dump(value, f)
            mlflow.log_artifact(cm_path)

def save_model(model_result: ModelResult, timestamp: str) -> None:
    """Save model with metadata"""
    model_path = model_result.model_path
    
    # Save the model
    model_file = model_path / "model.joblib"
    joblib.dump(model_result.model, model_file)
    mlflow.log_artifact(str(model_file))  # Log the model file as an artifact

    # Save metadata
    metadata = {
        "model_name": model_result.model_name,
        "timestamp": timestamp,
        "metrics": {
            "train_accuracy": model_result.metrics.train_accuracy,
            "val_accuracy": model_result.metrics.val_accuracy,
            "train_f1": model_result.metrics.train_f1,
            "val_f1": model_result.metrics.val_f1,
            "model_size": model_result.model_size
        }
    }

    metadata_file = model_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    mlflow.log_artifact(str(metadata_file))  # Log the metadata file as an artifact


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model_params: Dict[str, Any],
    m_dir: Path
) -> ModelResult:
    """
    Train a single model with MLflow tracking and comprehensive evaluation
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = m_dir / f"{model_name}_{timestamp}"
    model_path.mkdir(exist_ok=True, parents=True)
    
    # Start MLflow run with explicit run name
    with mlflow.start_run(run_name=f"{model_name}_{timestamp}", nested=True) as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("timestamp", timestamp)
        
        run_id = run.info.run_id
        print(f"{model_name} run ID: {run_id}")

        # Train model
        registry = ModelRegistry()
        model = registry.get_model(model_name, model_params)
        model.fit(X_train, y_train)

        # Evaluate on training set
        train_metrics = evaluate_model(model, X_train, y_train)
        log_metrics_to_mlflow(train_metrics, model_path, prefix="train_")

        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test)
        log_metrics_to_mlflow(test_metrics, model_path, prefix="val_")

        # Calculate and log model size
        model_size = get_model_size(model)
        mlflow.log_metric("model_size", model_size)

        # Log model using MLflow's built-in model logging
        mlflow.sklearn.log_model(
            model, 
            f"model_{model_name}",
            registered_model_name=f"{model_name}_model"
        )

        # Create ModelMetrics object
        metrics = ModelMetrics(
            train_accuracy=train_metrics['accuracy'],
            train_precision=train_metrics['precision'],
            train_recall=train_metrics['recall'],
            train_f1=train_metrics['f1'],
            train_roc_auc=train_metrics['roc_auc'],
            val_accuracy=test_metrics['accuracy'],
            val_precision=test_metrics['precision'],
            val_recall=test_metrics['recall'],
            val_f1=test_metrics['f1'],
            val_roc_auc=test_metrics['roc_auc']
        )

        result = ModelResult(model_name, model, metrics, model_size, model_path)
        save_model(result, timestamp)
        
        return result

def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    params_dict: Dict[str, Dict[str, Any]],
    experiment_name: str,
    dataset_name: str,
    encoderr: str,
    scalerr: str,
) -> Tuple[List[ModelResult], ModelResult, Path]:
    """
    Train all models and return results
    """
    # Create or set experiment
    mlflow.set_experiment(experiment_name)
    
    # Split data once for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    timestamp_mm = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    mlmr_dir = Path(f"mlruns/{dataset_name}_{timestamp_mm}")
    mlmr_dir.mkdir(exist_ok=True, parents=True)
    
    with mlflow.start_run(run_name=f"{dataset_name}_multiple_models_comparison") as multirun:
        # Log dataset characteristics
        mlflow.log_params({
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y)),
            "dataset_name": dataset_name
        })
        
        multirun_id = multirun.info.run_id
        print(f"MultiRun ID: {multirun_id}")

        # Save and log data files
        data_dir = mlmr_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        for name, data in [
            ("X_train", X_train), ("y_train", y_train),
            ("X_test", X_test), ("y_test", y_test)
        ]:
            file_path = data_dir / f"{name}.csv"
            data.to_csv(file_path, index=False)
            mlflow.log_artifact(str(file_path))

        # Save and log preprocessors
        pp_dir = mlmr_dir / "preprocessor"
        pp_dir.mkdir(exist_ok=True)
        
        for name, processor in [("encoder", encoderr), ("scaler", scalerr)]:
            file_path = pp_dir / f"{name}.joblib"
            joblib.dump(processor, file_path)
            mlflow.log_artifact(str(file_path))

        # Train models
        models_dir = mlmr_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, params in params_dict.items():
            result = train_model(
                X_train, X_test, y_train, y_test,
                model_name, params, models_dir
            )
            results.append(result)

    best_model = select_best_model(results)
    return results, best_model, best_model.model_path

def select_best_model(results: List[ModelResult]) -> ModelResult:
    """
    Select the best model based on validation metrics and model size
    Strategy:
    1. Find models with validation F1 score within 2% of the best F1 score
    2. Among those, select the one with smallest size
    """
    # Get best F1 score
    best_f1 = max(result.metrics.val_f1 for result in results)
    
    top_models = [
        result for result in results 
        if result.metrics.val_f1 >= best_f1
    ]
    
    # Among top models, select the smallest one
    return min(top_models, key=lambda x: x.model_size)