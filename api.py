from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from src.utils import preprocess_data

class PredictionInput(BaseModel):
    data: Dict[str, Any]


class PredictionResponse(BaseModel):
    predictions: List[float]
    timestamp: str


class ModelState:
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.encoder: Optional[Any] = None
        self.scaler: Optional[Any] = None

    def load_model(self):
        """Load model and preprocessors from config"""
        try:
            if os.path.exists("model_config.json"):
                with open("model_config.json", "r") as f:
                    config = json.load(f)
                    self.model_path = os.path.join(config["model_path"], "model.joblib")

                    if self.model_path and os.path.exists(self.model_path):
                        self.model = joblib.load(self.model_path)
                        # Load preprocessors from the same directory
                        encoder_path = os.path.join(
                            config["preprocessor_path"], "encoder.joblib"
                        )
                        scaler_path = os.path.join(
                            config["preprocessor_path"], "scaler.joblib"
                        )
                        if os.path.exists(encoder_path):
                            self.encoder = joblib.load(encoder_path)
                        if os.path.exists(scaler_path):
                            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.clear()

    def clear(self):
        """Clear model and preprocessors from memory"""
        self.model = None
        self.scaler = None
        self.encoder = None
        self.model_path = None

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data using loaded preprocessors"""
        # if self.target_column not in df.columns:
        #     # Add dummy target column for preprocessing
        #     df[self.target_column] = 0

        # Use the preprocessing function with loaded encoder and scaler
        X_processed, _, _, _ = preprocess_data(
            df=df,
            # target_column=self.target_column,
            # encoder=self.encoder,
            # scaler=self.scaler
        )

        return X_processed


# Initialize model state
model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading and unloading model"""
    # Startup: Load the model
    model_state.load_model()
    yield
    # Cleanup: Clear the model from memory
    model_state.clear()


app = FastAPI(title="Model Inference API", lifespan=lifespan)


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Endpoint for batch predictions on CSV file"""
    if not model_state.model:
        return {"error": "Model not loaded"}

    # Read CSV file
    df = pd.read_csv(file.file)

    try:
        # Preprocess data using loaded preprocessors
        X = model_state.preprocess_input(df)

        # Make predictions
        predictions = model_state.model.predict(X)

        # Create output DataFrame
        output_df = df.copy()
        output_df["prediction"] = predictions

        # Create predictions directory if it doesn't exist
        os.makedirs("predictions", exist_ok=True)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("predictions", f"predictions_{timestamp}.csv")
        output_df.to_csv(output_path, index=False)

        return {
            "message": "Predictions saved successfully",
            "file_path": output_path,
            "predictions_count": len(predictions),
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Endpoint for single prediction"""
    if not model_state.model:
        return {"error": "Model not loaded"}

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data.data])

        # Preprocess data using loaded preprocessors
        X = model_state.preprocess_input(df)

        # Make predictions
        predictions = model_state.model.predict(X).tolist()

        return PredictionResponse(
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.get("/health")
async def health_check():
    """Endpoint to check if the service is healthy and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": model_state.model is not None,
        "encoder_loaded": model_state.encoder is not None,
        "scaler_loaded": model_state.scaler is not None,
        "model_path": model_state.model_path,
    }