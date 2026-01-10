"""
Automated Model Retraining Pipeline
Retrain model with new data and deploy if performance improves
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Automated model retraining pipeline
    """

    def __init__(self, config_path: str = None):
        """
        Initialize retrainer

        Args:
            config_path: Path to config.yaml
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Model retrainer initialized")

    def load_new_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load new data for retraining

        Args:
            data_path: Path to new data CSV

        Returns:
            DataFrame with new data
        """
        if data_path is None:
            # Load from default location
            data_path = Path(__file__).parent.parent / self.config['data']['train_file']

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        return df

    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """
        Preprocess data for training

        Args:
            df: Raw dataframe

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        logger.info("Preprocessing data...")

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)

        # Feature engineering
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
                df['credit_utilization_bucket'] = df['credit_utilization_bucket'].cat.codes

        # Separate features and target
        target = self.config['model']['target_column']
        X = df.drop(columns=[target])
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle imbalanced data with SMOTE
        if self.config['imbalance']['strategy'] == 'SMOTE':
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(
                sampling_strategy=self.config['imbalance']['sampling_ratio'],
                random_state=self.config['model']['random_state']
            )
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        logger.info(f"Training set size: {X_train_scaled.shape}")
        logger.info(f"Test set size: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        logger.info("Training XGBoost model...")

        # Get hyperparameters from config
        params = self.config['xgboost']

        # Initialize model
        model = XGBClassifier(**params)

        # Train model
        model.fit(X_train, y_train)

        logger.info("Model training completed")

        return model

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Model Performance:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")

        return metrics

    def compare_with_current_model(self, new_metrics: dict) -> bool:
        """
        Compare new model with current production model

        Args:
            new_metrics: Metrics of newly trained model

        Returns:
            True if new model is better
        """
        # Load current model metrics
        current_metrics_path = self.models_dir / "current_model_metrics.json"

        if not current_metrics_path.exists():
            logger.info("No current model metrics found. New model will be deployed.")
            return True

        with open(current_metrics_path, 'r') as f:
            current_metrics = json.load(f)

        # Compare ROC-AUC (primary metric)
        improvement = new_metrics['roc_auc'] - current_metrics['roc_auc']

        logger.info(f"Current model ROC-AUC: {current_metrics['roc_auc']:.4f}")
        logger.info(f"New model ROC-AUC: {new_metrics['roc_auc']:.4f}")
        logger.info(f"Improvement: {improvement:.4f}")

        # Deploy if improvement is at least 1%
        if improvement >= 0.01:
            logger.info("New model shows significant improvement. Ready for deployment.")
            return True
        else:
            logger.info("New model does not show significant improvement. Keeping current model.")
            return False

    def deploy_model(
        self,
        model,
        scaler,
        metrics: dict,
        training_stats: dict = None
    ):
        """
        Deploy new model to production

        Args:
            model: Trained model
            scaler: Fitted scaler
            metrics: Model metrics
            training_stats: Training data statistics for drift detection
        """
        logger.info("Deploying new model...")

        # Backup current model
        current_model_path = self.models_dir / "production_model.pkl"
        if current_model_path.exists():
            backup_path = self.models_dir / f"production_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(joblib.load(current_model_path), backup_path)
            logger.info(f"Current model backed up to {backup_path}")

        # Save new model
        joblib.dump(model, current_model_path)
        logger.info(f"New model saved to {current_model_path}")

        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save metrics
        metrics_path = self.models_dir / "current_model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save training statistics for drift detection
        if training_stats:
            stats_path = self.models_dir / "training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(training_stats, f, indent=2)
            logger.info(f"Training statistics saved to {stats_path}")

        # Log deployment
        deployment_log_path = self.models_dir / "deployment_history.json"
        deployment_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': str(current_model_path)
        }

        if deployment_log_path.exists():
            with open(deployment_log_path, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append(deployment_entry)

        with open(deployment_log_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("Model deployment completed successfully")

    def calculate_training_stats(self, X: pd.DataFrame) -> dict:
        """
        Calculate training data statistics for drift detection

        Args:
            X: Training features

        Returns:
            Dictionary of statistics
        """
        stats = {}

        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                stats[f"{col}_mean"] = float(X[col].mean())
                stats[f"{col}_std"] = float(X[col].std())
                stats[f"{col}_min"] = float(X[col].min())
                stats[f"{col}_max"] = float(X[col].max())

        return stats

    def retrain_and_deploy(
        self,
        data_path: str = None,
        force_deploy: bool = False
    ) -> dict:
        """
        Complete retraining and deployment pipeline

        Args:
            data_path: Path to new training data
            force_deploy: Force deployment even if no improvement

        Returns:
            Dictionary with retraining results
        """
        logger.info("=" * 50)
        logger.info("Starting automated model retraining pipeline")
        logger.info("=" * 50)

        # Load data
        df = self.load_new_data(data_path)

        # Preprocess
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(df)

        # Calculate training statistics
        target = self.config['model']['target_column']
        X_features = df.drop(columns=[target])
        training_stats = self.calculate_training_stats(X_features)

        # Train model
        model = self.train_model(X_train, y_train)

        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)

        # Compare with current model
        should_deploy = force_deploy or self.compare_with_current_model(metrics)

        result = {
            'retrain_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'deployed': should_deploy
        }

        # Deploy if better
        if should_deploy:
            self.deploy_model(model, scaler, metrics, training_stats)
            result['deployment_status'] = 'success'
        else:
            result['deployment_status'] = 'skipped'

        logger.info("=" * 50)
        logger.info("Retraining pipeline completed")
        logger.info("=" * 50)

        return result


# Scheduled retraining function
def scheduled_retrain(config_path: str = None):
    """
    Function to be called by scheduler (e.g., cron job)

    Args:
        config_path: Path to config.yaml
    """
    retrainer = ModelRetrainer(config_path)
    result = retrainer.retrain_and_deploy()

    # Log result
    logger.info(f"Scheduled retraining completed: {result}")

    return result


# Example usage
if __name__ == "__main__":
    # Initialize retrainer
    retrainer = ModelRetrainer()

    # Run retraining pipeline
    result = retrainer.retrain_and_deploy(force_deploy=False)

    print("\n" + "=" * 50)
    print("RETRAINING RESULTS")
    print("=" * 50)
    print(json.dumps(result, indent=2))
