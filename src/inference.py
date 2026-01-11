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

        # Get expected feature names and order from the trained model
        if hasattr(self.model, 'feature_names_in_'):
            self.expected_features = list(self.model.feature_names_in_)
            logger.info(f"Model expects {len(self.expected_features)} features in specific order")
        else:
            self.expected_features = None
            logger.warning("Model doesn't have feature_names_in_ attribute")

        # Get features expected by scaler (should be a subset of model features)
        if hasattr(self.scaler, 'feature_names_in_'):
            self.scaler_features = list(self.scaler.feature_names_in_)
            logger.info(f"Scaler expects {len(self.scaler_features)} features")
        else:
            self.scaler_features = None
            logger.warning("Scaler doesn't have feature_names_in_ attribute")

        logger.info("Predictor initialized successfully")

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to raw data
        Must match EXACTLY the feature engineering from notebooks/02_feature_engineering.ipynb

        Args:
            data: Raw input dataframe

        Returns:
            Preprocessed dataframe with engineered features
        """
        df = data.copy()

        # 0. Add missing input features with default values
        # The model was trained on 10 features, but API might only send 4
        # Set reasonable defaults for missing features
        if 'NumberOfTime30-59DaysPastDueNotWorse' not in df.columns:
            df['NumberOfTime30-59DaysPastDueNotWorse'] = 0
        if 'NumberOfOpenCreditLinesAndLoans' not in df.columns:
            df['NumberOfOpenCreditLinesAndLoans'] = 0
        if 'NumberOfTimes90DaysLate' not in df.columns:
            df['NumberOfTimes90DaysLate'] = 0
        if 'NumberRealEstateLoansOrLines' not in df.columns:
            df['NumberRealEstateLoansOrLines'] = 0
        if 'NumberOfTime60-89DaysPastDueNotWorse' not in df.columns:
            df['NumberOfTime60-89DaysPastDueNotWorse'] = 0
        if 'NumberOfDependents' not in df.columns:
            df['NumberOfDependents'] = 0

        # 1. Create missing value indicator features (before imputation)
        if 'MonthlyIncome' in df.columns:
            df['MonthlyIncome_Missing'] = df['MonthlyIncome'].isnull().astype(int)

        if 'NumberOfDependents' in df.columns:
            df['NumberOfDependents_Missing'] = df['NumberOfDependents'].isnull().astype(int)

        # 2. Impute missing values (same as training)
        if 'MonthlyIncome' in df.columns:
            df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

        if 'NumberOfDependents' in df.columns:
            df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)

        # 3. Create derived features (matching training pipeline exactly)

        # Total Delinquencies
        if all(col in df.columns for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                                               'NumberOfTime60-89DaysPastDueNotWorse',
                                               'NumberOfTimes90DaysLate']):
            df['TotalDelinquencies'] = (
                df['NumberOfTime30-59DaysPastDueNotWorse'] +
                df['NumberOfTime60-89DaysPastDueNotWorse'] +
                df['NumberOfTimes90DaysLate']
            )

        # Severe Delinquency Flag
        if 'NumberOfTimes90DaysLate' in df.columns:
            df['HasSevereDelinquency'] = (df['NumberOfTimes90DaysLate'] > 0).astype(int)

        # Credit Utilization Buckets (0: Low, 1: Medium, 2: High)
        if 'RevolvingUtilizationOfUnsecuredLines' in df.columns:
            def categorize_utilization(util):
                if util <= 0.30:
                    return 0  # Low
                elif util <= 0.70:
                    return 1  # Medium
                else:
                    return 2  # High

            df['UtilizationBucket'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(categorize_utilization)

        # Age Groups (0: Young, 1: Middle, 2: Senior)
        if 'age' in df.columns:
            def categorize_age(age):
                if age < 35:
                    return 0  # Young
                elif age < 55:
                    return 1  # Middle
                else:
                    return 2  # Senior

            df['AgeGroup'] = df['age'].apply(categorize_age)

        # Income-to-Debt Ratio
        if 'MonthlyIncome' in df.columns and 'DebtRatio' in df.columns:
            df['IncomeToDebtRatio'] = np.where(
                df['DebtRatio'] > 0,
                df['MonthlyIncome'] / df['DebtRatio'],
                df['MonthlyIncome']  # If no debt, just use income
            )
            # Cap at 99th percentile (use a reasonable max value)
            df['IncomeToDebtRatio'] = df['IncomeToDebtRatio'].clip(upper=3563869.0)

        # Utilization Per Credit Line
        if 'RevolvingUtilizationOfUnsecuredLines' in df.columns and 'NumberOfOpenCreditLinesAndLoans' in df.columns:
            df['UtilizationPerCreditLine'] = np.where(
                df['NumberOfOpenCreditLinesAndLoans'] > 0,
                df['RevolvingUtilizationOfUnsecuredLines'] / df['NumberOfOpenCreditLinesAndLoans'],
                0
            )

        # Has Real Estate Flag
        if 'NumberRealEstateLoansOrLines' in df.columns:
            df['HasRealEstate'] = (df['NumberRealEstateLoansOrLines'] > 0).astype(int)

        # Debt-to-Income Ratio
        if 'DebtRatio' in df.columns and 'MonthlyIncome' in df.columns:
            df['DebtToIncomeRatio'] = df['DebtRatio'] * df['MonthlyIncome']
            # Cap at 99th percentile (use a reasonable max value)
            df['DebtToIncomeRatio'] = df['DebtToIncomeRatio'].clip(upper=26384400.0)

        # 4. Handle outliers (same as training)
        outlier_features = [
            'RevolvingUtilizationOfUnsecuredLines',
            'DebtRatio',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]

        for feature in outlier_features:
            if feature in df.columns:
                # Use the same percentile caps as training
                if feature == 'RevolvingUtilizationOfUnsecuredLines':
                    df[feature] = df[feature].clip(lower=0.00, upper=1.09)
                elif feature == 'DebtRatio':
                    df[feature] = df[feature].clip(lower=0.00, upper=4979.04)
                elif feature == 'NumberOfTime30-59DaysPastDueNotWorse':
                    df[feature] = df[feature].clip(lower=0.00, upper=4.00)
                elif feature == 'NumberOfTime60-89DaysPastDueNotWorse':
                    df[feature] = df[feature].clip(lower=0.00, upper=2.00)
                elif feature == 'NumberOfTimes90DaysLate':
                    df[feature] = df[feature].clip(lower=0.00, upper=3.00)

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

        # Prepare features for the model
        if self.expected_features is not None:
            # Ensure all expected features are present
            missing_features = [f for f in self.expected_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Select features in the correct order
            X_all = df[self.expected_features]
        else:
            # Fallback: select all features except target
            feature_cols = [col for col in df.columns if col != self.target_column]
            X_all = df[feature_cols]

        # Scale only the features the scaler expects (continuous features)
        # The scaler was trained on a subset of features (excluding binary/categorical ones)
        if self.scaler_features is not None:
            X_to_scale = X_all[self.scaler_features]
            X_scaled_subset = pd.DataFrame(
                self.scaler.transform(X_to_scale),
                columns=self.scaler_features,
                index=X_all.index
            )

            # Replace the scaled features in the full feature set
            X_final = X_all.copy()
            X_final[self.scaler_features] = X_scaled_subset
        else:
            # Fallback: scale all features
            X_final = pd.DataFrame(
                self.scaler.transform(X_all),
                columns=X_all.columns,
                index=X_all.index
            )

        # Predict probabilities using all features
        probas = self.model.predict_proba(X_final)

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