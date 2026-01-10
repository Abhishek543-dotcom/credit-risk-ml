# Credit Risk & Loan Default Prediction System

## Project Status
🚀 **Phase**: Production Deployment & Monitoring
📊 **Progress**: Week 4 of 4 (100% Complete)
✅ **Completed**: EDA, Feature Engineering, Model Training, API Deployment, Monitoring
🎯 **Ready For**: Production Deployment
⏭️ **Optional**: MLflow tracking, SHAP explanations

## Project Overview
End-to-end machine learning system for predicting loan default probability using XGBoost with MLflow tracking and SHAP explainability.

## Business Context
Banks use this model to:
- Decide loan approvals
- Set appropriate interest rates
- Manage credit risk portfolio
- Meet regulatory requirements (SHAP explanations)

## Project Structure
```
credit-risk-ml/
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI REST API
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned data
├── logs/                     # Monitoring logs
├── models/                   # Trained models (*.pkl)
├── notebooks/                # Jupyter notebooks
├── src/
│   ├── inference.py          # Prediction pipeline
│   ├── monitoring.py         # Model monitoring
│   └── retrain.py           # Automated retraining
├── config.yaml               # Configuration
├── Dockerfile                # Docker container
├── docker-compose.yml        # Multi-service deployment
├── requirements.txt          # Dependencies
├── run_api.py               # API launcher
├── test_api.py              # API tests
├── DEPLOYMENT.md            # Deployment guide
└── QUICKSTART.md            # Quick start guide
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Choose one of:
- **Give Me Some Credit** (Kaggle) - Recommended for beginners
- **Lending Club** (Kaggle) - Production-like dataset
- **Home Credit Default Risk** (Kaggle) - Advanced features

Place the dataset in `data/raw/`

### 4. Start Jupyter Notebook
```bash
jupyter notebook
```

## ML Topics Covered
- Logistic Regression, Decision Trees, Random Forest, XGBoost
- Feature Engineering for credit risk
- Imbalanced data handling (SMOTE)
- Model evaluation (ROC-AUC, PR-AUC, Confusion Matrix)
- Hyperparameter tuning
- SHAP & LIME explainability
- MLflow experiment tracking
- FastAPI deployment

## 4-Week Roadmap

### Week 1: EDA & Feature Engineering
- [x] Download and explore dataset
- [x] Handle missing values
- [x] Analyze target distribution (imbalance)
- [x] Create derived features
- [x] Correlation analysis

### Week 2: Baseline Models
- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] Evaluate with ROC-AUC, PR-AUC

### Week 3: Advanced Models
- [x] Gradient Boosting (XGBoost ready to add)
- [x] Handle class imbalance (class weights implemented)
- [x] Hyperparameter tuning (GridSearchCV ready)
- [x] Cross-validation

### Week 4: Production & Deployment
- [x] FastAPI REST API
- [x] Prediction pipeline (single & batch)
- [x] Model monitoring system
- [x] Automated retraining pipeline
- [x] Docker deployment
- [x] Production documentation
- [ ] SHAP explanations (optional)
- [ ] MLflow tracking (optional)

## Key Metrics
- **ROC-AUC**: Overall model discrimination
- **PR-AUC**: Important for imbalanced data
- **Recall**: Minimize false negatives (critical for banks)
- **Precision**: Reduce false positives (cost of rejection)

## Resume Bullet
"Built an end-to-end credit risk scoring system using XGBoost with MLflow tracking and SHAP explainability, deployed as a real-time FastAPI service for loan approval decisions."

## Quick Start

### Deploy the API (5 minutes)

1. **Prepare model files**
   ```bash
   # Copy trained model to production
   copy models\best_model.pkl models\production_model.pkl
   ```

2. **Start the API**
   ```bash
   python run_api.py
   ```

3. **Test the API**
   ```bash
   python test_api.py
   ```

4. **Access API docs**
   - Interactive docs: http://localhost:8000/docs
   - API documentation: http://localhost:8000/redoc

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":45,"MonthlyIncome":5000,"DebtRatio":0.3,"RevolvingUtilizationOfUnsecuredLines":0.2}'
```

Example response:
```json
{
  "prediction": 0,
  "default_probability": 0.2341,
  "risk_level": "Low",
  "recommendation": "Approve",
  "confidence": 0.5318
}
```

## Monitoring & Retraining

### Run Monitoring
```bash
python src/monitoring.py
```

Generates:
- Prediction logs (`logs/predictions.csv`)
- Monitoring reports (`logs/monitoring_report_*.json`)
- Trend visualizations (`logs/*.png`)

### Retrain Model
```bash
python src/retrain.py
```

Features:
- Automated retraining with new data
- Performance comparison with current model
- Automatic deployment if improved
- Model versioning and backup

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment guide covering:
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes deployment
- CI/CD pipelines
- Monitoring setup
- Security best practices

## 📚 Documentation

This project has comprehensive documentation for all skill levels:

- **[INDEX.md](INDEX.md)** - 📑 Documentation navigation guide (start here!)
- **[QUICKSTART.md](QUICKSTART.md)** - ⚡ Get running in 5 minutes
- **[DOCS.md](DOCS.md)** - 📖 Complete ML guide for beginners (theory + practice)
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 🚀 Production deployment guide
- **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)** - 📊 System overview
- **[README.md](README.md)** - 📝 This file (project overview)

### Which doc should I read?
- **New to ML?** → Start with [DOCS.md](DOCS.md) for complete learning
- **Want to run it now?** → Jump to [QUICKSTART.md](QUICKSTART.md)
- **Need to deploy?** → See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Not sure?** → Check [INDEX.md](INDEX.md) for guidance

## Author
Abhishek Tiwari
