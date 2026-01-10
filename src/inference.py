"""
Credit Risk Prediction Pipeline
Loads trained model and makes predictions on new data
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Credit Risk Prediction Pipeline
    Loads model, preprocessor, and makes predictions
    """

    def __init__(self, model_path: str = None, scaler_path: str = None, config_path: str = None):
        """
        Initialize predictor with model and scaler

        Args:
            model_path: Path to trained model (.pkl)
            scaler_path: Path to fitted scaler (.pkl)
            config_path: Path to config.yaml
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set default paths from config
        if model_path is None:
            model_path = Path(__file__).parent.parent / self.config['api']['model_path']
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"

        # Load model and scaler
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        logger.info(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Get feature info from config
        self.numeric_features = self.config['features']['numeric_features']
        self.target_column = self.config['model']['target_column']

        logger.info("Predictor initialized successfully")

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to raw data

        Args:
            data: Raw input dataframe

        Returns:
            Preprocessed dataframe with engineered features
        """
        df = data.copy()

        # Create derived features (matching training pipeline)
        if 'debt_to_income_ratio' in self.config['features']['derived_features']:
            if 'DebtRatio' in df.columns and 'MonthlyIncome' in df.columns:
                df['debt_to_income_ratio'] = df['DebtRatio'] * df['MonthlyIncome']

        if 'credit_utilization_bucket' in self.config['features']['derived_features']:
            if 'RevolvingUtilizationOfUnsecuredLines' in df.columns:
                df['credit_utilization_bucket'] = pd.cut(
                    df['RevolvingUtilizationOfUnsecuredLines'],
                    bins=[0, 0.3, 0.7, 1.0, float('inf')],
                    labels=['low', 'medium', 'high', 'very_high']
                )
                # Convert to numeric
                df['credit_utilization_bucket'] = df['credit_utilization_bucket'].cat.codes

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)

        return df

    def predict_proba(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Predict default probabilities

        Args:
            data: Input data (DataFrame, dict, or list of dicts)

        Returns:
            Array of default probabilities
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Preprocess features
        df = self.preprocess_features(df)

        # Select features used in training
        feature_cols = [col for col in df.columns if col != self.target_column]
        X = df[feature_cols]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probas = self.model.predict_proba(X_scaled)

        # Return probability of default (class 1)
        return probas[:, 1]

    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]], threshold: float = 0.5) -> np.ndarray:
        """
        Predict default classes

        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            threshold: Classification threshold (default 0.5)

        Returns:
            Array of predicted classes (0 or 1)
        """
        probas = self.predict_proba(data)
        return (probas >= threshold).astype(int)

    def predict_with_details(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Predict with detailed output including probability and risk level

        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            threshold: Classification threshold (default 0.5)

        Returns:
            List of prediction dictionaries with details
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Get predictions
        probas = self.predict_proba(df)
        predictions = self.predict(df, threshold)

        # Create detailed results
        results = []
        for i in range(len(df)):
            prob = float(probas[i])
            pred = int(predictions[i])

            # Determine risk level
            if prob < 0.3:
                risk_level = "Low"
            elif prob < 0.5:
                risk_level = "Medium"
            elif prob < 0.7:
                risk_level = "High"
            else:
                risk_level = "Very High"

            result = {
                "prediction": pred,
                "default_probability": round(prob, 4),
                "risk_level": risk_level,
                "recommendation": "Reject" if pred == 1 else "Approve",
                "confidence": round(abs(prob - 0.5) * 2, 4)  # Distance from decision boundary
            }
            results.append(result)

        return results


def load_predictor(model_path: str = None) -> CreditRiskPredictor:
    """
    Convenience function to load predictor

    Args:
        model_path: Optional path to model file

    Returns:
        Initialized CreditRiskPredictor
    """
    return CreditRiskPredictor(model_path=model_path)


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = CreditRiskPredictor()

    # Example single prediction
    sample_data = {
        'age': 45,
        'MonthlyIncome': 5000,
        'DebtRatio': 0.3,
        'RevolvingUtilizationOfUnsecuredLines': 0.2
    }

    # Get detailed prediction
    result = predictor.predict_with_details(sample_data)
    print("\n=== Single Prediction ===")
    print(result[0])

    # Example batch prediction
    batch_data = [
        {'age': 45, 'MonthlyIncome': 5000, 'DebtRatio': 0.3, 'RevolvingUtilizationOfUnsecuredLines': 0.2},
        {'age': 30, 'MonthlyIncome': 3000, 'DebtRatio': 0.8, 'RevolvingUtilizationOfUnsecuredLines': 0.9},
    ]

    results = predictor.predict_with_details(batch_data)
    print("\n=== Batch Predictions ===")
    for i, result in enumerate(results):
        print(f"Sample {i+1}: {result}")