"""
Quick test script to verify the fix works
Run this after restarting the API server
"""

import requests
import json

# Test the prediction endpoint
data = {
    "age": 45,
    "MonthlyIncome": 5000,
    "DebtRatio": 0.3,
    "RevolvingUtilizationOfUnsecuredLines": 0.2
}

print("Testing API with sample data:")
print(json.dumps(data, indent=2))
print("\nSending request...")

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=data,
        timeout=10
    )

    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS! Prediction received:")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Default Probability: {result['default_probability']:.4f}")
        print(f"  - Risk Level: {result['risk_level']}")
        print(f"  - Recommendation: {result['recommendation']}")
        print(f"  - Confidence: {result['confidence']:.4f}")
    else:
        print(f"\n❌ FAILED: Status code {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
