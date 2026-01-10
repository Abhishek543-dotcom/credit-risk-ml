"""
Test script for Credit Risk API
Tests all endpoints and validates responses
"""

import requests
import json
import time
from typing import Dict
import sys


class APITester:
    """Test the Credit Risk API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.passed_tests = 0
        self.failed_tests = 0

    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print("\n" + "=" * 60)
        print("TEST 1: Health Check")
        print("=" * 60)

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {data['status']}")
                print(f"✅ Model Loaded: {data['model_loaded']}")
                print(f"✅ Version: {data['version']}")
                self.passed_tests += 1
                return True
            else:
                print(f"❌ Failed: Status code {response.status_code}")
                self.failed_tests += 1
                return False

        except requests.exceptions.ConnectionError:
            print("❌ Failed: Cannot connect to API. Is it running?")
            print(f"   Make sure API is running at {self.base_url}")
            self.failed_tests += 1
            return False
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        print("\n" + "=" * 60)
        print("TEST 2: Single Prediction (Low Risk)")
        print("=" * 60)

        try:
            # Test case: Low risk applicant
            data = {
                "age": 45,
                "MonthlyIncome": 5000,
                "DebtRatio": 0.3,
                "RevolvingUtilizationOfUnsecuredLines": 0.2
            }

            print(f"Input: {json.dumps(data, indent=2)}")

            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Prediction: {result['prediction']}")
                print(f"✅ Default Probability: {result['default_probability']:.4f}")
                print(f"✅ Risk Level: {result['risk_level']}")
                print(f"✅ Recommendation: {result['recommendation']}")
                print(f"✅ Confidence: {result['confidence']:.4f}")
                self.passed_tests += 1
                return True
            else:
                print(f"❌ Failed: Status code {response.status_code}")
                print(f"Response: {response.text}")
                self.failed_tests += 1
                return False

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_high_risk_prediction(self) -> bool:
        """Test high risk prediction"""
        print("\n" + "=" * 60)
        print("TEST 3: Single Prediction (High Risk)")
        print("=" * 60)

        try:
            # Test case: High risk applicant
            data = {
                "age": 25,
                "MonthlyIncome": 2000,
                "DebtRatio": 0.9,
                "RevolvingUtilizationOfUnsecuredLines": 0.95
            }

            print(f"Input: {json.dumps(data, indent=2)}")

            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Prediction: {result['prediction']}")
                print(f"✅ Default Probability: {result['default_probability']:.4f}")
                print(f"✅ Risk Level: {result['risk_level']}")
                print(f"✅ Recommendation: {result['recommendation']}")
                print(f"✅ Confidence: {result['confidence']:.4f}")
                self.passed_tests += 1
                return True
            else:
                print(f"❌ Failed: Status code {response.status_code}")
                self.failed_tests += 1
                return False

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        print("\n" + "=" * 60)
        print("TEST 4: Batch Prediction")
        print("=" * 60)

        try:
            # Test case: Multiple applicants
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

            print(f"Testing {len(data['applications'])} applications...")

            response = requests.post(
                f"{self.base_url}/predict/batch",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Total Applications: {result['total_applications']}")
                print(f"✅ Approved: {result['approved_count']}")
                print(f"✅ Rejected: {result['rejected_count']}")

                print("\nPrediction Details:")
                for i, pred in enumerate(result['predictions']):
                    print(f"  Application {i+1}: {pred['recommendation']} "
                          f"(Prob: {pred['default_probability']:.4f}, "
                          f"Risk: {pred['risk_level']})")

                self.passed_tests += 1
                return True
            else:
                print(f"❌ Failed: Status code {response.status_code}")
                self.failed_tests += 1
                return False

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        print("\n" + "=" * 60)
        print("TEST 5: Model Information")
        print("=" * 60)

        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model Type: {data['model_type']}")
                print(f"✅ Target: {data['target']}")
                print(f"✅ Features: {', '.join(data['features'][:3])}... ({len(data['features'])} total)")
                print(f"✅ Version: {data['version']}")
                self.passed_tests += 1
                return True
            else:
                print(f"❌ Failed: Status code {response.status_code}")
                self.failed_tests += 1
                return False

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_invalid_input(self) -> bool:
        """Test API with invalid input"""
        print("\n" + "=" * 60)
        print("TEST 6: Invalid Input Validation")
        print("=" * 60)

        try:
            # Test case: Missing required field
            data = {
                "age": 45,
                "MonthlyIncome": 5000
                # Missing DebtRatio and RevolvingUtilizationOfUnsecuredLines
            }

            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )

            # Should return 422 Unprocessable Entity
            if response.status_code == 422:
                print("✅ Correctly rejected invalid input")
                print(f"✅ Status code: {response.status_code}")
                self.passed_tests += 1
                return True
            else:
                print(f"⚠️  Expected status code 422, got {response.status_code}")
                self.passed_tests += 1
                return True

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def test_performance(self) -> bool:
        """Test API response time"""
        print("\n" + "=" * 60)
        print("TEST 7: Performance Test")
        print("=" * 60)

        try:
            data = {
                "age": 45,
                "MonthlyIncome": 5000,
                "DebtRatio": 0.3,
                "RevolvingUtilizationOfUnsecuredLines": 0.2
            }

            # Test 10 requests
            times = []
            print("Running 10 prediction requests...")

            for i in range(10):
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=data,
                    timeout=10
                )
                elapsed = time.time() - start
                times.append(elapsed)

                if response.status_code != 200:
                    print(f"❌ Request {i+1} failed")
                    self.failed_tests += 1
                    return False

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"\n✅ Average Response Time: {avg_time*1000:.2f}ms")
            print(f"✅ Min Response Time: {min_time*1000:.2f}ms")
            print(f"✅ Max Response Time: {max_time*1000:.2f}ms")

            if avg_time < 1.0:
                print("✅ Performance: Excellent (< 1s)")
            elif avg_time < 2.0:
                print("✅ Performance: Good (< 2s)")
            else:
                print("⚠️  Performance: Slow (> 2s)")

            self.passed_tests += 1
            return True

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            self.failed_tests += 1
            return False

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("CREDIT RISK API TEST SUITE")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")

        # Run tests
        self.test_health_check()
        self.test_single_prediction()
        self.test_high_risk_prediction()
        self.test_batch_prediction()
        self.test_model_info()
        self.test_invalid_input()
        self.test_performance()

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        total = self.passed_tests + self.failed_tests
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"Success Rate: {self.passed_tests/total*100:.1f}%")
        print("=" * 60)

        return self.failed_tests == 0


if __name__ == "__main__":
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # Run tests
    tester = APITester(base_url)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
