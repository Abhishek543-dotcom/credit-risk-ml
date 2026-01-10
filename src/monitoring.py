"""
Model Monitoring System
Tracks model performance, data drift, and prediction distribution
"""

import pandas as pd
import numpy as np
import joblib
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance and data quality in production
    """

    def __init__(self, config_path: str = None, log_dir: str = None):
        """
        Initialize model monitor

        Args:
            config_path: Path to config.yaml
            log_dir: Directory to store monitoring logs
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set log directory
        if log_dir is None:
            self.log_dir = Path(__file__).parent.parent / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Initialize prediction log
        self.prediction_log_path = self.log_dir / "predictions.csv"

        # Load training statistics for drift detection
        self.training_stats_path = Path(__file__).parent.parent / "models" / "training_stats.json"

        logger.info("Model monitor initialized")

    def log_prediction(
        self,
        features: Dict,
        prediction: int,
        probability: float,
        application_id: str = None
    ):
        """
        Log a prediction for monitoring

        Args:
            features: Input features
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            application_id: Optional application identifier
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'application_id': application_id or datetime.now().strftime('%Y%m%d%H%M%S'),
            'prediction': prediction,
            'probability': probability,
            **features
        }

        # Convert to DataFrame
        df = pd.DataFrame([log_entry])

        # Append to log file
        if self.prediction_log_path.exists():
            df.to_csv(self.prediction_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.prediction_log_path, mode='w', header=True, index=False)

        logger.info(f"Logged prediction for application {application_id}")

    def load_predictions(self, days: int = 7) -> pd.DataFrame:
        """
        Load recent predictions

        Args:
            days: Number of days to load

        Returns:
            DataFrame of predictions
        """
        if not self.prediction_log_path.exists():
            logger.warning("No prediction log found")
            return pd.DataFrame()

        # Load all predictions
        df = pd.read_csv(self.prediction_log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]

        return df

    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Calculate model performance metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'timestamp': datetime.now().isoformat()
        }

        return metrics

    def check_performance_degradation(
        self,
        current_metrics: Dict,
        baseline_metrics: Dict,
        threshold: float = 0.05
    ) -> Tuple[bool, List[str]]:
        """
        Check if model performance has degraded

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline (training) metrics
            threshold: Acceptable degradation threshold (5% default)

        Returns:
            Tuple of (is_degraded, list of degraded metrics)
        """
        degraded_metrics = []

        for metric in ['roc_auc', 'precision', 'recall', 'f1_score']:
            if metric in current_metrics and metric in baseline_metrics:
                current = current_metrics[metric]
                baseline = baseline_metrics[metric]

                degradation = baseline - current

                if degradation > threshold:
                    degraded_metrics.append(f"{metric}: {baseline:.3f} -> {current:.3f} (Δ{degradation:.3f})")
                    logger.warning(f"Performance degradation detected in {metric}")

        return len(degraded_metrics) > 0, degraded_metrics

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        training_stats: Dict = None,
        threshold: float = 0.1
    ) -> Tuple[bool, List[str]]:
        """
        Detect data drift in feature distributions

        Args:
            current_data: Current production data
            training_stats: Training data statistics
            threshold: Drift threshold (10% default)

        Returns:
            Tuple of (has_drift, list of drifted features)
        """
        if training_stats is None:
            if not self.training_stats_path.exists():
                logger.warning("No training statistics found for drift detection")
                return False, []

            with open(self.training_stats_path, 'r') as f:
                training_stats = json.load(f)

        drifted_features = []

        for feature in current_data.columns:
            if feature in training_stats and feature not in ['timestamp', 'application_id', 'prediction', 'probability']:
                # Calculate mean difference
                current_mean = current_data[feature].mean()
                training_mean = training_stats.get(f"{feature}_mean", current_mean)

                # Calculate percentage difference
                if training_mean != 0:
                    drift = abs((current_mean - training_mean) / training_mean)

                    if drift > threshold:
                        drifted_features.append(
                            f"{feature}: {training_mean:.3f} -> {current_mean:.3f} (Δ{drift*100:.1f}%)"
                        )
                        logger.warning(f"Data drift detected in {feature}")

        return len(drifted_features) > 0, drifted_features

    def analyze_prediction_distribution(self, days: int = 7) -> Dict:
        """
        Analyze prediction distribution over time

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with distribution statistics
        """
        df = self.load_predictions(days=days)

        if df.empty:
            return {}

        analysis = {
            'total_predictions': len(df),
            'approval_rate': (df['prediction'] == 0).mean(),
            'rejection_rate': (df['prediction'] == 1).mean(),
            'avg_default_probability': df['probability'].mean(),
            'std_default_probability': df['probability'].std(),
            'high_risk_count': (df['probability'] > 0.7).sum(),
            'low_risk_count': (df['probability'] < 0.3).sum(),
            'timestamp': datetime.now().isoformat()
        }

        return analysis

    def generate_monitoring_report(self, days: int = 7) -> Dict:
        """
        Generate comprehensive monitoring report

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with monitoring report
        """
        logger.info(f"Generating monitoring report for last {days} days")

        # Load predictions
        df = self.load_predictions(days=days)

        if df.empty:
            return {
                'status': 'No data',
                'message': 'No predictions found in the specified time period'
            }

        # Prediction distribution
        distribution = self.analyze_prediction_distribution(days=days)

        # Data drift check
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'application_id', 'prediction', 'probability']]
        has_drift, drifted_features = self.detect_data_drift(df[feature_cols])

        # Create report
        report = {
            'report_date': datetime.now().isoformat(),
            'monitoring_period_days': days,
            'prediction_distribution': distribution,
            'data_quality': {
                'has_drift': has_drift,
                'drifted_features': drifted_features
            },
            'alerts': []
        }

        # Add alerts
        if has_drift:
            report['alerts'].append({
                'severity': 'WARNING',
                'message': 'Data drift detected',
                'details': drifted_features
            })

        if distribution.get('rejection_rate', 0) > 0.5:
            report['alerts'].append({
                'severity': 'INFO',
                'message': f"High rejection rate: {distribution['rejection_rate']*100:.1f}%"
            })

        # Save report
        report_path = self.log_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {report_path}")

        return report

    def plot_prediction_trends(self, days: int = 7, save_path: str = None):
        """
        Plot prediction trends over time

        Args:
            days: Number of days to plot
            save_path: Path to save plot
        """
        df = self.load_predictions(days=days)

        if df.empty:
            logger.warning("No data to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Predictions - Last {days} Days', fontsize=16)

        # 1. Predictions over time
        df['date'] = df['timestamp'].dt.date
        daily_predictions = df.groupby('date').agg({
            'prediction': ['sum', 'count']
        })
        daily_predictions.columns = ['rejected', 'total']
        daily_predictions['approved'] = daily_predictions['total'] - daily_predictions['rejected']

        axes[0, 0].plot(daily_predictions.index, daily_predictions['total'], marker='o', label='Total')
        axes[0, 0].plot(daily_predictions.index, daily_predictions['approved'], marker='o', label='Approved')
        axes[0, 0].plot(daily_predictions.index, daily_predictions['rejected'], marker='o', label='Rejected')
        axes[0, 0].set_title('Daily Predictions')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Probability distribution
        axes[0, 1].hist(df['probability'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Default Probability Distribution')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Approval rate over time
        daily_approval_rate = daily_predictions['approved'] / daily_predictions['total'] * 100
        axes[1, 0].plot(daily_approval_rate.index, daily_approval_rate.values, marker='o', color='green')
        axes[1, 0].set_title('Daily Approval Rate')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Approval Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Risk level distribution
        df['risk_level'] = pd.cut(
            df['probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        risk_counts = df['risk_level'].value_counts()
        axes[1, 1].bar(risk_counts.index, risk_counts.values, color=['green', 'yellow', 'orange', 'red'])
        axes[1, 1].set_title('Risk Level Distribution')
        axes[1, 1].set_xlabel('Risk Level')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            save_path = self.log_dir / f"prediction_trends_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = ModelMonitor()

    # Simulate some predictions
    print("Simulating predictions...")
    for i in range(100):
        features = {
            'age': np.random.randint(25, 65),
            'MonthlyIncome': np.random.randint(2000, 10000),
            'DebtRatio': np.random.uniform(0.1, 0.8),
            'RevolvingUtilizationOfUnsecuredLines': np.random.uniform(0.1, 0.9)
        }
        prediction = np.random.choice([0, 1], p=[0.7, 0.3])
        probability = np.random.uniform(0.1, 0.9)

        monitor.log_prediction(features, prediction, probability, f"APP_{i:04d}")

    # Generate monitoring report
    print("\nGenerating monitoring report...")
    report = monitor.generate_monitoring_report(days=7)
    print(json.dumps(report, indent=2))

    # Plot trends
    print("\nGenerating prediction trends plot...")
    monitor.plot_prediction_trends(days=7)
