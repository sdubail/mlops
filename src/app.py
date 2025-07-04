"""FastAPI application for our ML model"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import MLModel
from .schemas import HealthResponse, PredictionRequest, PredictionResponse

# Configuration
app = FastAPI(
    title="MLOps Demo API",
    description="Demonstration API for ML deployment on Cloud Run",
    version="0.1.0",
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model at startup
model = MLModel()


@app.on_event("startup")
async def startup_event():
    """Load model at API startup"""
    try:
        model.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # In production, you might want to fail fast here
        # raise e


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint - simple health check"""
    return {"message": "MLOps Demo API", "status": "running", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check for Cloud Run"""
    try:
        # Check if model is loaded
        is_model_ready = model.is_ready()

        return HealthResponse(
            status="healthy" if is_model_ready else "unhealthy",
            model_loaded=is_model_ready,
            version="0.1.0",
            environment=os.getenv("ENV", "development"),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main endpoint for predictions"""
    try:
        # Check if model is ready
        if not model.is_ready():
            raise HTTPException(
                status_code=503, detail="Model not ready. Please try again later."
            )

        # Make prediction
        prediction = model.predict(request.features)

        # Calculate confidence (dummy example)
        confidence = model.get_prediction_confidence(request.features)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=model.get_version(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Information about the current model"""
    if not model.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": model.get_model_type(),
        "version": model.get_version(),
        "features_expected": model.get_expected_features(),
        "trained_at": model.get_training_date(),
    }


# For running locally
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
