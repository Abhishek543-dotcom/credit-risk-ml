# Quick Start Guide - Credit Risk ML API

Get your Credit Risk ML model deployed and running in 5 minutes!

## Prerequisites
- Python 3.10+ installed
- Trained model files in `models/` directory
- Virtual environment activated

## Step 1: Verify Model Files (30 seconds)

Check that you have the required model files:

```bash
ls models/
```

You should see:
- `best_model.pkl` (or `production_model.pkl`)
- `scaler.pkl`

If `production_model.pkl` doesn't exist, copy from `best_model.pkl`:

```bash
# Windows
copy models\best_model.pkl models\production_model.pkl

# Linux/Mac
cp models/best_model.pkl models/production_model.pkl
```

## Step 2: Start the API (1 minute)

```bash
# Make sure you're in the project root
cd credit-risk-ml

# Activate virtual environment (if not already activated)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Start the API
python run_api.py
```

You should see:
```
============================================================
Starting Credit Risk ML API
============================================================
Host: 0.0.0.0
Port: 8000
Docs: http://0.0.0.0:8000/docs
ReDoc: http://0.0.0.0:8000/redoc
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 3: Test the API (2 minutes)

### Option A: Interactive API Docs

Open your browser and go to:
```
http://localhost:8000/docs
```

Try the `/predict` endpoint:
1. Click "Try it out"
2. Use the example data provided
3. Click "Execute"
4. See the prediction result!

### Option B: Test Script

Open a new terminal and run:

```bash
python test_api.py
```

This will run 7 automated tests and show you the results.

### Option C: Command Line (curl)

```bash
# Windows (PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -ContentType "application/json" -Body '{"age":45,"MonthlyIncome":5000,"DebtRatio":0.3,"RevolvingUtilizationOfUnsecuredLines":0.2}'

# Linux/Mac
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":45,"MonthlyIncome":5000,"DebtRatio":0.3,"RevolvingUtilizationOfUnsecuredLines":0.2}'
```

### Option D: Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 45,
        "MonthlyIncome": 5000,
        "DebtRatio": 0.3,
        "RevolvingUtilizationOfUnsecuredLines": 0.2
    }
)

print(response.json())
```

## Expected Output

```json
{
  "prediction": 0,
  "default_probability": 0.2341,
  "risk_level": "Low",
  "recommendation": "Approve",
  "confidence": 0.5318
}
```

## Step 4: Test Monitoring (1 minute)

```bash
# Run monitoring (generates sample predictions and creates report)
python src/monitoring.py
```

Check the generated files:
```bash
# View prediction log
cat logs/predictions.csv

# View monitoring report
ls logs/monitoring_report_*.json

# View plots
ls logs/*.png
```

## Step 5: Test Inference Pipeline (30 seconds)

Test the prediction pipeline standalone:

```bash
python src/inference.py
```

## Common Issues

### Issue: Port already in use
```bash
# Use a different port
python run_api.py --port 8080
```

### Issue: Model not found
```bash
# Copy best model to production
copy models\best_model.pkl models\production_model.pkl
```

### Issue: Module not found
```bash
# Make sure you're in the project root
cd credit-risk-ml

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

Now that your API is running:

1. **Read the full documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Set up monitoring**: Configure automated monitoring (see DEPLOYMENT.md)
3. **Configure retraining**: Set up automated retraining (see DEPLOYMENT.md)
4. **Deploy to production**: Use Docker or cloud platforms (see DEPLOYMENT.md)

## Useful Commands

```bash
# Start API with auto-reload (for development)
python run_api.py --reload

# Start API on different port
python run_api.py --port 8080

# Test the API
python test_api.py

# Run monitoring
python src/monitoring.py

# Retrain model
python src/retrain.py

# Test inference pipeline
python src/inference.py
```

## API Endpoints

- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Predict (Single)**: POST http://localhost:8000/predict
- **Predict (Batch)**: POST http://localhost:8000/predict/batch
- **Model Info**: GET http://localhost:8000/model/info

## Support

Need help? Check:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Full deployment guide
- [README.md](README.md) - Project overview
- GitHub Issues - Report problems

---

**You're ready to go! 🚀**

Your Credit Risk ML model is now deployed and ready to make predictions!
