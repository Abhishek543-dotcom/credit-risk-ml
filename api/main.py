"""
Credit Risk Prediction API
FastAPI service for real-time loan default predictions
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import sys
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import CreditRiskPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting loan default probability using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loaded once at startup)
predictor = None


# Pydantic models for request/response validation
class LoanApplication(BaseModel):
    """Single loan application data"""
    age: float = Field(..., ge=18, le=100, description="Applicant age in years")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income in dollars")
    DebtRatio: float = Field(..., ge=0, description="Debt-to-income ratio")
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, description="Credit utilization ratio"
    )

    @validator('age', 'MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines')
    def check_not_null(cls, v):
        if v is None:
            raise ValueError('Field cannot be null')
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "MonthlyIncome": 5000,
                "DebtRatio": 0.3,
                "RevolvingUtilizationOfUnsecuredLines": 0.2
            }
        }


class BatchLoanApplications(BaseModel):
    """Batch of loan applications"""
    applications: List[LoanApplication]

    class Config:
        schema_extra = {
            "example": {
                "applications": [
                    {
                        "age": 45,
                        "MonthlyIncome": 5000,
                        "DebtRatio": 0.3,
                        "RevolvingUtilizationOfUnsecuredLines": 0.2
                    },
                    {
                        "age": 30,
                        "MonthlyIncome": 3000,
                        "DebtRatio": 0.8,
                        "RevolvingUtilizationOfUnsecuredLines": 0.9
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: int = Field(..., description="Predicted class (0=no default, 1=default)")
    default_probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High, Very High")
    recommendation: str = Field(..., description="Loan decision recommendation")
    confidence: float = Field(..., description="Model confidence (0-1)")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_applications: int
    approved_count: int
    rejected_count: int


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    try:
        logger.info("Loading model...")
        predictor = CreditRiskPredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(application: LoanApplication):
    """
    Predict default probability for a single loan application

    Args:
        application: Loan application data

    Returns:
        Prediction with probability, risk level, and recommendation
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert to dict
        data = application.dict()

        # Get prediction
        result = predictor.predict_with_details(data)[0]

        logger.info(f"Prediction made: {result['recommendation']}")

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(batch: BatchLoanApplications):
    """
    Predict default probabilities for multiple loan applications

    Args:
        batch: Batch of loan applications

    Returns:
        List of predictions with summary statistics
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert to list of dicts
        data = [app.dict() for app in batch.applications]

        # Get predictions
        results = predictor.predict_with_details(data)

        # Calculate summary stats
        approved = sum(1 for r in results if r['prediction'] == 0)
        rejected = sum(1 for r in results if r['prediction'] == 1)

        logger.info(f"Batch prediction: {len(results)} applications, {approved} approved, {rejected} rejected")

        return {
            "predictions": results,
            "total_applications": len(results),
            "approved_count": approved,
            "rejected_count": rejected
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get model information and configuration"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_type": type(predictor.model).__name__,
        "features": predictor.numeric_features + predictor.config['features']['derived_features'],
        "target": predictor.target_column,
        "model_path": str(Path(__file__).parent.parent / config['api']['model_path']),
        "version": "1.0.0"
    }


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    # Get config
    host = config['api']['host']
    port = config['api']['port']

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
