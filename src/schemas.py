"""Pydantic schemas for input and output data validation"""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Schema for prediction requests"""

    features: list[float] = Field(description="List of features for prediction")

    @field_validator("features")
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("At least one feature is required")
        if len(v) > 100:  # Reasonable limit
            raise ValueError("Too many features (max 100)")
        return v


class PredictionResponse(BaseModel):
    """Schema for prediction responses"""

    prediction: float = Field(..., description="Prediction result")
    confidence: float = Field(
        ..., description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )
    model_version: str = Field(..., description="Version of the model used")


class HealthResponse(BaseModel):
    """Schema for health check responses"""

    status: str = Field(description="Service status")

    model_loaded: bool = Field(..., description="Indicates if the model is loaded")
    version: str = Field(..., description="API version")
    environment: str = Field(
        default="development", description="Deployment environment"
    )


class ErrorResponse(BaseModel):
    """Schema for error responses"""

    detail: str = Field(..., description="Error description")
    error_code: str | None = Field(None, description="Specific error code")
