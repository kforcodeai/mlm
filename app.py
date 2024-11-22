import streamlit as st
import pandas as pd
from src.train import train_all_models, ModelMetrics, ModelResult
from src.utils import preprocess_data
from src.models import ModelRegistry
import subprocess
import json
import os
import time
import requests
from typing import Dict, Any, Tuple, List

def create_param_input(model_name: str, param_name: str, param_config: Dict[str, Any]) -> Any:
    """Create appropriate Streamlit input widget based on parameter type with unique keys"""
    # Create a unique key by combining model name and parameter name
    widget_key = f"{model_name}_{param_name}"
    
    if param_config['type'] == 'int':
        return st.slider(
            f"{param_name}",
            min_value=param_config['min'],
            max_value=param_config['max'],
            value=param_config['min'],
            step=param_config['step'],
            key=widget_key
        )
    elif param_config['type'] == 'float':
        return st.slider(
            f"{param_name}",
            min_value=float(param_config['min']),
            max_value=float(param_config['max']),
            value=float(param_config['min']),
            step=float(param_config['step']),
            key=widget_key
        )
    elif param_config['type'] == 'categorical':
        return st.selectbox(
            f"{param_name}", 
            param_config['options'],
            key=widget_key
        )

def display_metrics_table(metrics: ModelMetrics, prefix: str) -> None:
    """Display a formatted metrics table"""
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Value": [
            getattr(metrics, f"{prefix}_accuracy"),
            getattr(metrics, f"{prefix}_precision"),
            getattr(metrics, f"{prefix}_recall"),
            getattr(metrics, f"{prefix}_f1"),
            getattr(metrics, f"{prefix}_roc_auc")
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def check_api_status():
    """Check if the FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/docs")
        return response.status_code == 200
    except:
        return False

def save_deployment_config(model_path: str, preprocessor_path:str):
    # Save model path to config file
    config = {
        "model_path": model_path,
        "preprocessor_path": preprocessor_path
    }
    with open("model_config.json", "w") as f:
        json.dump(config, f)

def deploy_model() -> bool:
    """Deploy model by saving its path to config and starting FastAPI server"""
    try:
        # Check if API is already running
        if not check_api_status():
            # Start the API server
            subprocess.Popen(
                ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE
            )
            
            # Wait for API to start (with timeout)
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 seconds timeout
                if check_api_status():
                    return True
                time.sleep(1)
            return False
        return True
    except Exception as e:
        st.error(f"Failed to deploy model: {str(e)}")
        return False

def main():
    
    st.title("ML Model Training Dashboard")
    # Initialize session state
    if 'model_deployed' not in st.session_state:
        st.session_state.model_deployed = False
    if 'deploy_clicked' not in st.session_state:
        st.session_state.deploy_clicked = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = ""
    
    registry = ModelRegistry()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv", key="dataset_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Select target column
        target_column = st.selectbox("Select target column", df.columns, key="target_column")
        st.session_state.dataset_name = st.text_input("Dataset name", key="dataset_name_input")
        
        # Create tabs for all models
        model_names = list(registry.models.keys())
        tabs = st.tabs([registry.models[name].name for name in model_names])
        
        # Dictionary to store parameters for each model
        model_params = {}
        
        # Create parameter inputs for each model
        for model_name, tab in zip(model_names, tabs):
            model_config = registry.models[model_name]
            
            with tab:
                st.subheader(f"{model_config.name} Parameters")
                model_params[model_name] = {}
                
                cols = st.columns(2)
                for i, (param_name, param_config) in enumerate(model_config.param_ranges.items()):
                    with cols[i % 2]:
                        model_params[model_name][param_name] = create_param_input(
                            model_name,
                            param_name,
                            param_config
                        )
        
        # Training button
        if st.button("Train All Models", key="train_button"):
            X, y, enc, sclr = preprocess_data(df, target_column)

            with st.spinner("Training models..."):
                # Train all models
                results, best_model, best_model_path = train_all_models(
                    X, y,
                    model_params,
                    experiment_name=f"multi_model_comparison_{st.session_state.dataset_name}",
                    dataset_name=st.session_state.dataset_name,
                    encoderr=enc, scalerr=sclr
                )
                preprocessor_path = os.path.join('/'.join(str(best_model_path).split('/')[:-2]), 'preprocessor')
                save_deployment_config(str(best_model_path), str(preprocessor_path))
                # Store results in session state
                st.session_state.training_completed = True
                st.session_state.best_model = best_model
                st.session_state.results = results
                
                # Display results
                st.success("Models trained successfully!")
                st.write(f'Dataset trained: {st.session_state.dataset_name}')
                
                results_df = pd.DataFrame([
                    {
                        "Model": registry.models[result.model_name].name,
                        "Training Accuracy": f"{result.metrics.train_accuracy:.4f}",
                        "Validation Accuracy": f"{result.metrics.val_accuracy:.4f}",
                        "Training F1": f"{result.metrics.train_f1:.4f}",
                        "Validation F1": f"{result.metrics.val_f1:.4f}",
                        "Size (bytes)": f"{result.model_size:,}",
                        "Best Model": "✓" if result == best_model else ""
                    }
                    for result in results
                ])
                
                # Display results in an expander
                with st.expander("Model Comparison Results", expanded=True):
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                    )
                    
                    st.subheader("Best Model Details")
                    st.info(f"Recommended model: {registry.models[best_model.model_name].name}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Training Metrics")
                        display_metrics_table(best_model.metrics, "train")
                    
                    with col2:
                        st.subheader("Validation Metrics")
                        display_metrics_table(best_model.metrics, "val")
        
        # Model Deployment Section
        if st.session_state.training_completed:
            st.markdown("---")
            st.header("Model Deployment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Selected Model",
                    registry.models[st.session_state.best_model.model_name].name
                )
                st.metric(
                    "Validation F1 Score",
                    f"{st.session_state.best_model.metrics.val_f1:.4f}"
                )
            
            with col2:
                if not st.session_state.model_deployed:
                    if st.button("Deploy Model", key="deploy_button"):
                        st.session_state.deploy_clicked = True
                        
                        with st.spinner("Deploying model..."):
                            if deploy_model():
                                st.session_state.model_deployed = True
                                st.success("Model deployed successfully!")
                                st.markdown("""
                                ### API Endpoints:
                                
                                1. **File-based prediction**: 
                                   - URL: `http://localhost:8000/predict/file`
                                   - Method: POST
                                   - Upload a CSV file for batch predictions
                                
                                2. **Single prediction**:
                                   - URL: `http://localhost:8000/predict`
                                   - Method: POST
                                   - Send JSON data for individual predictions
                                
                                API Documentation: [Swagger UI](http://localhost:8000/docs)
                                """)
                            else:
                                st.error("Failed to deploy model. Please check the logs.")
                else:
                    st.success("Model is currently deployed")
                    if st.button("Redeploy Model", key="redeploy_button"):
                        st.session_state.model_deployed = False
                        st.rerun()

        # Display MLflow UI link
        st.markdown(
            "View detailed metrics in [MLflow Dashboard](http://localhost:8080)"
        )

main()