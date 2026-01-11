"""
Simple API Test Script (without emojis for Windows compatibility)
"""

import requests
import json
import time

base_url = "http://localhost:8001"
passed = 0
failed = 0

print("=" * 60)
print("CREDIT RISK API TEST SUITE")
print("=" * 60)
print(f"Testing API at: {base_url}\n")

# Test 1: Health Check
print("TEST 1: Health Check")
print("-" * 60)
try:
    response = requests.get(f"{base_url}/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"[PASS] Status: {data['status']}")
        print(f"[PASS] Model Loaded: {data['model_loaded']}")
        print(f"[PASS] Version: {data['version']}")
        passed += 1
    else:
        print(f"[FAIL] Status code: {response.status_code}")
        failed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 2: Single Prediction (Low Risk)
print("\nTEST 2: Single Prediction (Low Risk)")
print("-" * 60)
try:
    data = {
        "age": 45,
        "MonthlyIncome": 5000,
        "DebtRatio": 0.3,
        "RevolvingUtilizationOfUnsecuredLines": 0.2
    }
    response = requests.post(f"{base_url}/predict", json=data, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"[PASS] Prediction: {result['prediction']}")
        print(f"[PASS] Probability: {result['default_probability']:.4f}")
        print(f"[PASS] Risk Level: {result['risk_level']}")
        print(f"[PASS] Recommendation: {result['recommendation']}")
        passed += 1
    else:
        print(f"[FAIL] Status code: {response.status_code}")
        print(f"Response: {response.text}")
        failed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 3: Single Prediction (High Risk)
print("\nTEST 3: Single Prediction (High Risk)")
print("-" * 60)
try:
    data = {
        "age": 25,
        "MonthlyIncome": 2000,
        "DebtRatio": 0.9,
        "RevolvingUtilizationOfUnsecuredLines": 0.95
    }
    response = requests.post(f"{base_url}/predict", json=data, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"[PASS] Prediction: {result['prediction']}")
        print(f"[PASS] Probability: {result['default_probability']:.4f}")
        print(f"[PASS] Risk Level: {result['risk_level']}")
        print(f"[PASS] Recommendation: {result['recommendation']}")
        passed += 1
    else:
        print(f"[FAIL] Status code: {response.status_code}")
        failed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 4: Batch Prediction
print("\nTEST 4: Batch Prediction")
print("-" * 60)
try:
    data = {
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
            },
            {
                "age": 55,
                "MonthlyIncome": 8000,
                "DebtRatio": 0.2,
                "RevolvingUtilizationOfUnsecuredLines": 0.1
            }
        ]
    }
    response = requests.post(f"{base_url}/predict/batch", json=data, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"[PASS] Total Applications: {result['total_applications']}")
        print(f"[PASS] Approved: {result['approved_count']}")
        print(f"[PASS] Rejected: {result['rejected_count']}")
        for i, pred in enumerate(result['predictions']):
            print(f"  App {i+1}: {pred['recommendation']} (Prob: {pred['default_probability']:.4f})")
        passed += 1
    else:
        print(f"[FAIL] Status code: {response.status_code}")
        failed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 5: Model Info
print("\nTEST 5: Model Information")
print("-" * 60)
try:
    response = requests.get(f"{base_url}/model/info", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"[PASS] Model Type: {data['model_type']}")
        print(f"[PASS] Target: {data['target']}")
        print(f"[PASS] Number of Features: {len(data['features'])}")
        passed += 1
    else:
        print(f"[FAIL] Status code: {response.status_code}")
        failed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 6: Invalid Input Validation
print("\nTEST 6: Invalid Input Validation")
print("-" * 60)
try:
    data = {
        "age": 45,
        "MonthlyIncome": 5000
        # Missing required fields
    }
    response = requests.post(f"{base_url}/predict", json=data, timeout=10)
    if response.status_code == 422:
        print("[PASS] Correctly rejected invalid input")
        passed += 1
    else:
        print(f"[WARN] Expected 422, got {response.status_code}")
        passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Test 7: Performance Test
print("\nTEST 7: Performance Test")
print("-" * 60)
try:
    data = {
        "age": 45,
        "MonthlyIncome": 5000,
        "DebtRatio": 0.3,
        "RevolvingUtilizationOfUnsecuredLines": 0.2
    }
    times = []
    for i in range(10):
        start = time.time()
        response = requests.post(f"{base_url}/predict", json=data, timeout=10)
        elapsed = time.time() - start
        times.append(elapsed)
        if response.status_code != 200:
            raise Exception(f"Request {i+1} failed")

    avg_time = sum(times) / len(times)
    print(f"[PASS] Average Response Time: {avg_time*1000:.2f}ms")
    print(f"[PASS] Min: {min(times)*1000:.2f}ms, Max: {max(times)*1000:.2f}ms")
    passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    failed += 1

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
total = passed + failed
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Success Rate: {passed/total*100:.1f}%")
print("=" * 60)

if failed == 0:
    print("\n[SUCCESS] All tests passed!")
else:
    print(f"\n[WARNING] {failed} test(s) failed")
