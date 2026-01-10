# Credit Risk ML - Deployment Guide

Complete guide for deploying the Credit Risk ML model to production.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [API Usage](#api-usage)
6. [Monitoring](#monitoring)
7. [Model Retraining](#model-retraining)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Trained model files in `models/` directory

### 1. Install Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Start the API
```bash
# Simple start
python run_api.py

# With custom host/port
python run_api.py --host 0.0.0.0 --port 8080

# With auto-reload for development
python run_api.py --reload
```

### 3. Access the API
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## Local Development

### Running the API Locally

**Method 1: Using run_api.py**
```bash
python run_api.py --reload
```

**Method 2: Using uvicorn directly**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Running the main module**
```bash
python -m api.main
```

### Testing the Prediction Pipeline

Test the inference pipeline standalone:
```bash
python src/inference.py
```

### Testing the API

Use the provided test script:
```bash
python test_api.py
```

Or use curl:
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "MonthlyIncome": 5000,
    "DebtRatio": 0.3,
    "RevolvingUtilizationOfUnsecuredLines": 0.2
  }'
```

---

## Docker Deployment

### Build and Run with Docker

**1. Build the Docker image**
```bash
docker build -t credit-risk-api:latest .
```

**2. Run the container**
```bash
docker run -d \
  --name credit-risk-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  credit-risk-api:latest
```

**3. Check logs**
```bash
docker logs -f credit-risk-api
```

### Using Docker Compose

**1. Start all services**
```bash
docker-compose up -d
```

**2. View logs**
```bash
docker-compose logs -f api
```

**3. Stop services**
```bash
docker-compose down
```

**Services included:**
- `api`: Credit Risk API (port 8000)
- `mlflow`: MLflow tracking server (port 5000)

---

## Production Deployment

### Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with production settings:
```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MODEL_PATH=models/production_model.pkl
PREDICTION_THRESHOLD=0.5
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up HTTPS/SSL certificates
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Configure monitoring and alerts
- [ ] Set up automated backups
- [ ] Test disaster recovery
- [ ] Document API access credentials
- [ ] Set up API rate limiting
- [ ] Configure CORS policies

### Deployment Options

#### Option 1: Cloud Platforms (AWS, GCP, Azure)

**AWS EC2 Deployment:**
```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose -y

# 4. Clone repository
git clone your-repo-url
cd credit-risk-ml

# 5. Deploy with Docker Compose
sudo docker-compose up -d
```

**AWS Elastic Beanstalk:**
```bash
# Initialize EB CLI
eb init -p docker credit-risk-api

# Create environment and deploy
eb create production-env
eb deploy
```

#### Option 2: Kubernetes

See `kubernetes/` directory for K8s manifests.

#### Option 3: Serverless (AWS Lambda)

See `serverless/` directory for configuration.

---

## API Usage

### Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-10T10:30:00",
  "version": "1.0.0"
}
```

#### 2. Single Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
  "age": 45,
  "MonthlyIncome": 5000,
  "DebtRatio": 0.3,
  "RevolvingUtilizationOfUnsecuredLines": 0.2
}
```

**Response:**
```json
{
  "prediction": 0,
  "default_probability": 0.2341,
  "risk_level": "Low",
  "recommendation": "Approve",
  "confidence": 0.5318
}
```

#### 3. Batch Prediction
```bash
POST /predict/batch
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "predictions": [...],
  "total_applications": 2,
  "approved_count": 1,
  "rejected_count": 1
}
```

#### 4. Model Information
```bash
GET /model/info
```

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Loan application data
data = {
    "age": 45,
    "MonthlyIncome": 5000,
    "DebtRatio": 0.3,
    "RevolvingUtilizationOfUnsecuredLines": 0.2
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"Recommendation: {result['recommendation']}")
print(f"Default Probability: {result['default_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## Monitoring

### Setting Up Monitoring

The monitoring system tracks:
- Prediction distribution
- Model performance
- Data drift
- API health

### Generate Monitoring Report

```bash
python src/monitoring.py
```

### View Monitoring Logs

```bash
# View recent predictions
cat logs/predictions.csv

# View monitoring reports
cat logs/monitoring_report_*.json

# View plots
ls logs/*.png
```

### Automated Monitoring

Set up a cron job (Linux/Mac):
```bash
# Edit crontab
crontab -e

# Add daily monitoring at 2 AM
0 2 * * * cd /path/to/credit-risk-ml && /path/to/venv/bin/python src/monitoring.py
```

Windows Task Scheduler:
```powershell
# Create scheduled task
schtasks /create /tn "Credit Risk Monitoring" /tr "python C:\path\to\credit-risk-ml\src\monitoring.py" /sc daily /st 02:00
```

### Monitoring Alerts

Configure alerts for:
- Performance degradation (ROC-AUC drop > 5%)
- Data drift detection
- High rejection rates (> 50%)
- API errors and downtime

---

## Model Retraining

### Manual Retraining

```bash
python src/retrain.py
```

The retraining pipeline will:
1. Load new training data
2. Preprocess and engineer features
3. Train new model
4. Evaluate performance
5. Compare with current model
6. Deploy if improved (ROC-AUC improvement > 1%)

### Automated Retraining

**Option 1: Cron Job (Linux/Mac)**
```bash
# Edit crontab
crontab -e

# Add weekly retraining every Sunday at 2 AM
0 2 * * 0 cd /path/to/credit-risk-ml && /path/to/venv/bin/python src/retrain.py
```

**Option 2: Windows Task Scheduler**
```powershell
schtasks /create /tn "Model Retraining" /tr "python C:\path\to\credit-risk-ml\src\retrain.py" /sc weekly /d SUN /st 02:00
```

**Option 3: Airflow DAG**

Create an Airflow DAG for scheduled retraining (see `airflow/dags/` directory).

### Retraining Best Practices

1. **Schedule**: Weekly or monthly depending on data volume
2. **Validation**: Always validate on hold-out test set
3. **Deployment**: Only deploy if improvement > 1% ROC-AUC
4. **Backup**: Previous models are automatically backed up
5. **Monitoring**: Monitor performance after deployment

---

## Troubleshooting

### Common Issues

#### Issue: Model not found
```
Error: Model file not found at models/production_model.pkl
```

**Solution:**
```bash
# Check if model files exist
ls models/

# If missing, copy from best_model.pkl
cp models/best_model.pkl models/production_model.pkl
```

#### Issue: Port already in use
```
Error: Address already in use
```

**Solution:**
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000

# Linux/Mac:
lsof -i :8000

# Kill the process or use different port
python run_api.py --port 8080
```

#### Issue: Import errors
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Make sure you're in the project root
cd credit-risk-ml

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: Prediction errors
```
Error: Feature mismatch
```

**Solution:**
- Ensure input data has all required features
- Check feature names match exactly (case-sensitive)
- Verify numeric types (no strings in numeric fields)

### Logs

Check logs for debugging:
```bash
# API logs (if using docker-compose)
docker-compose logs -f api

# Monitoring logs
cat logs/predictions.csv
cat logs/monitoring_report_*.json

# Application logs (if running locally)
# Logs are printed to console
```

### Performance Optimization

**For high-traffic scenarios:**

1. **Use multiple workers:**
```bash
python run_api.py --workers 4
```

2. **Enable caching:**
- Add Redis for response caching
- Cache model in memory (already done)

3. **Load balancing:**
- Use nginx or AWS ELB
- Deploy multiple instances

4. **Optimize model inference:**
- Use batch predictions when possible
- Consider model quantization

---

## Support

For issues or questions:
- Create an issue on GitHub
- Email: [your-email@example.com]
- Documentation: See README.md

---

## Next Steps

1. ✅ Deploy API locally
2. ✅ Test with sample predictions
3. ⬜ Set up monitoring
4. ⬜ Configure automated retraining
5. ⬜ Deploy to production
6. ⬜ Set up CI/CD pipeline

---

**Deployed by:** Abhishek Tiwari
**Last Updated:** 2024-01-10
