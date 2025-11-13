"""
Comprehensive API Test Suite

Tests all endpoints of the Hybrid ESN-Ridge Prediction API.

Usage:
    python test_api.py
    
    Or with specific test:
    python test_api.py --test test_health_check
"""
import sys
import os
import time
import argparse
from typing import Dict, Any, List
import requests
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# API base URL
BASE_URL = "http://localhost:8000"

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_test(name: str, status: str, message: str = ""):
    """Print test result"""
    status_symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
    color = "\033[92m" if status == "PASS" else "\033[91m" if status == "FAIL" else "\033[93m"
    reset = "\033[0m"
    
    print(f"{color}{status_symbol} {name}{reset}", end="")
    if message:
        print(f" - {message}")
    else:
        print()
    
    if status == "PASS":
        test_results["passed"].append(name)
    elif status == "FAIL":
        test_results["failed"].append(name)
    else:
        test_results["warnings"].append(name)


def check_server_running() -> bool:
    """Check if API server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_health_check():
    """Test GET / endpoint"""
    print_header("Test 1: Health Check (GET /)")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        
        # Check status code
        if response.status_code != 200:
            print_test("Health Check - Status Code", "FAIL", 
                      f"Expected 200, got {response.status_code}")
            return False
        
        print_test("Health Check - Status Code", "PASS")
        
        # Check response structure
        data = response.json()
        required_keys = ["status", "service", "version"]
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print_test("Health Check - Response Structure", "FAIL",
                      f"Missing keys: {missing_keys}")
            return False
        
        print_test("Health Check - Response Structure", "PASS")
        
        # Check response values
        if data["status"] != "online":
            print_test("Health Check - Status Value", "FAIL",
                      f"Expected 'online', got '{data['status']}'")
            return False
        
        print_test("Health Check - Status Value", "PASS")
        print_test("Health Check - Service Name", "PASS", 
                  f"Service: {data['service']}")
        print_test("Health Check - Version", "PASS", 
                  f"Version: {data['version']}")
        
        return True
        
    except requests.exceptions.Timeout:
        print_test("Health Check", "FAIL", "Request timeout")
        return False
    except requests.exceptions.ConnectionError:
        print_test("Health Check", "FAIL", "Connection error - is server running?")
        return False
    except Exception as e:
        print_test("Health Check", "FAIL", f"Unexpected error: {str(e)}")
        return False


def test_models_info():
    """Test GET /models/info endpoint"""
    print_header("Test 2: Models Info (GET /models/info)")
    
    try:
        response = requests.get(f"{BASE_URL}/models/info", timeout=10)
        
        # Check status code
        if response.status_code != 200:
            print_test("Models Info - Status Code", "FAIL",
                      f"Expected 200, got {response.status_code}")
            return False
        
        print_test("Models Info - Status Code", "PASS")
        
        # Check response structure
        data = response.json()
        required_keys = ["models", "status"]
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print_test("Models Info - Response Structure", "FAIL",
                      f"Missing keys: {missing_keys}")
            return False
        
        print_test("Models Info - Response Structure", "PASS")
        
        # Check status
        if data["status"] != "ready":
            print_test("Models Info - Status", "WARN",
                      f"Status is '{data['status']}', expected 'ready'")
        else:
            print_test("Models Info - Status", "PASS")
        
        # Check models structure
        models = data["models"]
        expected_horizons = ["h1", "h5", "h20"]
        missing_horizons = [h for h in expected_horizons if h not in models]
        
        if missing_horizons:
            print_test("Models Info - Model Horizons", "FAIL",
                      f"Missing horizons: {missing_horizons}")
            return False
        
        print_test("Models Info - Model Horizons", "PASS", 
                  f"Found: {', '.join(expected_horizons)}")
        
        # Check each model structure
        for horizon in expected_horizons:
            model_info = models[horizon]
            required_model_keys = ["fold", "sharpe"]
            missing_model_keys = [k for k in required_model_keys if k not in model_info]
            
            if missing_model_keys:
                print_test(f"Models Info - {horizon} Structure", "FAIL",
                          f"Missing keys: {missing_model_keys}")
                return False
            
            print_test(f"Models Info - {horizon}", "PASS",
                      f"fold_{model_info['fold']}, Sharpe: {model_info['sharpe']}")
        
        return True
        
    except requests.exceptions.Timeout:
        print_test("Models Info", "FAIL", "Request timeout")
        return False
    except Exception as e:
        print_test("Models Info", "FAIL", f"Unexpected error: {str(e)}")
        return False


def test_predict():
    """Test GET /predict endpoint"""
    print_header("Test 3: Predictions (GET /predict)")
    
    try:
        print("Fetching predictions (this may take 10-20 seconds)...")
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/predict", timeout=60)
        elapsed_time = time.time() - start_time
        
        # Check status code
        if response.status_code != 200:
            print_test("Predict - Status Code", "FAIL",
                      f"Expected 200, got {response.status_code}")
            if response.status_code == 500:
                try:
                    error_data = response.json()
                    print(f"  Error detail: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"  Response: {response.text[:200]}")
            return False
        
        print_test("Predict - Status Code", "PASS", 
                  f"Response time: {elapsed_time:.2f}s")
        
        # Check response structure
        data = response.json()
        required_keys = ["symbol", "predictions", "recent_news", "generated_at"]
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print_test("Predict - Response Structure", "FAIL",
                      f"Missing keys: {missing_keys}")
            return False
        
        print_test("Predict - Response Structure", "PASS")
        
        # Check symbol
        if data["symbol"] != "SPY":
            print_test("Predict - Symbol", "WARN",
                      f"Expected 'SPY', got '{data['symbol']}'")
        else:
            print_test("Predict - Symbol", "PASS")
        
        # Check predictions
        predictions = data["predictions"]
        if not isinstance(predictions, list):
            print_test("Predict - Predictions Type", "FAIL",
                      f"Expected list, got {type(predictions)}")
            return False
        
        print_test("Predict - Predictions Type", "PASS")
        
        if len(predictions) == 0:
            print_test("Predict - Predictions Count", "WARN",
                      "No predictions returned")
        else:
            print_test("Predict - Predictions Count", "PASS",
                      f"Found {len(predictions)} predictions")
        
        # Check prediction structure
        if len(predictions) > 0:
            first_pred = predictions[0]
            required_pred_keys = [
                "date", "h1_prediction", "h1_signal", "h5_prediction",
                "h5_signal", "h20_prediction", "h20_signal", "actual_close"
            ]
            missing_pred_keys = [k for k in required_pred_keys if k not in first_pred]
            
            if missing_pred_keys:
                print_test("Predict - Prediction Structure", "FAIL",
                          f"Missing keys: {missing_pred_keys}")
                return False
            
            print_test("Predict - Prediction Structure", "PASS")
            
            # Validate prediction values
            for key in ["h1_prediction", "h5_prediction", "h20_prediction", "actual_close"]:
                if not isinstance(first_pred[key], (int, float)):
                    print_test(f"Predict - {key} Type", "FAIL",
                              f"Expected number, got {type(first_pred[key])}")
                    return False
            
            print_test("Predict - Prediction Values Type", "PASS")
            
            # Check signals
            for horizon in ["h1", "h5", "h20"]:
                signal = first_pred[f"{horizon}_signal"]
                if signal not in ["BUY", "SELL"]:
                    print_test(f"Predict - {horizon} Signal", "WARN",
                              f"Expected BUY/SELL, got '{signal}'")
                else:
                    print_test(f"Predict - {horizon} Signal", "PASS",
                              f"Latest: {signal}")
            
            # Check date format
            try:
                datetime.strptime(first_pred["date"], "%Y-%m-%d")
                print_test("Predict - Date Format", "PASS")
            except ValueError:
                print_test("Predict - Date Format", "FAIL",
                          f"Invalid date format: {first_pred['date']}")
                return False
        
        # Check recent_news
        recent_news = data["recent_news"]
        if not isinstance(recent_news, list):
            print_test("Predict - News Type", "FAIL",
                      f"Expected list, got {type(recent_news)}")
            return False
        
        print_test("Predict - News Type", "PASS")
        print_test("Predict - News Count", "PASS",
                  f"Found {len(recent_news)} news items")
        
        # Check news structure
        if len(recent_news) > 0:
            first_news = recent_news[0]
            required_news_keys = ["date", "title", "publisher", "link"]
            missing_news_keys = [k for k in required_news_keys if k not in first_news]
            
            if missing_news_keys:
                print_test("Predict - News Structure", "WARN",
                          f"Missing keys: {missing_news_keys}")
            else:
                print_test("Predict - News Structure", "PASS")
        
        # Check generated_at
        try:
            datetime.fromisoformat(data["generated_at"].replace('Z', '+00:00'))
            print_test("Predict - Generated At Format", "PASS")
        except ValueError:
            print_test("Predict - Generated At Format", "WARN",
                      f"Invalid ISO format: {data['generated_at']}")
        
        return True
        
    except requests.exceptions.Timeout:
        print_test("Predict", "FAIL", "Request timeout (60s)")
        return False
    except Exception as e:
        print_test("Predict", "FAIL", f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_news():
    """Test GET /news endpoint with various parameters"""
    print_header("Test 4: News Headlines (GET /news)")
    
    test_cases = [
        {"params": {}, "name": "Default (7 days)"},
        {"params": {"days_back": 7}, "name": "7 days"},
        {"params": {"days_back": 30}, "name": "30 days"},
        {"params": {"days_back": 1}, "name": "1 day"},
        {"params": {"days_back": 90}, "name": "90 days"},
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        params = test_case["params"]
        name = test_case["name"]
        
        try:
            response = requests.get(f"{BASE_URL}/news", params=params, timeout=10)
            
            if response.status_code != 200:
                print_test(f"News - {name} Status", "FAIL",
                          f"Expected 200, got {response.status_code}")
                all_passed = False
                continue
            
            data = response.json()
            
            if not isinstance(data, list):
                print_test(f"News - {name} Type", "FAIL",
                          f"Expected list, got {type(data)}")
                all_passed = False
                continue
            
            print_test(f"News - {name}", "PASS",
                      f"Found {len(data)} headlines")
            
            # Check structure if news items exist
            if len(data) > 0:
                first_item = data[0]
                required_keys = ["date", "title", "publisher", "link"]
                missing_keys = [k for k in required_keys if k not in first_item]
                
                if missing_keys:
                    print_test(f"News - {name} Structure", "WARN",
                              f"Missing keys: {missing_keys}")
                else:
                    print_test(f"News - {name} Structure", "PASS")
            
        except requests.exceptions.Timeout:
            print_test(f"News - {name}", "FAIL", "Request timeout")
            all_passed = False
        except Exception as e:
            print_test(f"News - {name}", "FAIL", f"Error: {str(e)}")
            all_passed = False
    
    # Test invalid parameter (negative)
    # FastAPI Query validation returns 422 Unprocessable Entity
    try:
        response = requests.get(f"{BASE_URL}/news", params={"days_back": -1}, timeout=10)
        if response.status_code == 422:
            print_test("News - Invalid Parameter (negative)", "PASS",
                      "Correctly rejected negative days_back (422)")
        elif response.status_code == 400:
            print_test("News - Invalid Parameter (negative)", "PASS",
                      "Correctly rejected negative days_back (400)")
        elif response.status_code == 200:
            print_test("News - Invalid Parameter (negative)", "FAIL",
                      "Should reject negative days_back (got 200)")
            all_passed = False
        else:
            print_test("News - Invalid Parameter (negative)", "WARN",
                      f"Unexpected status code: {response.status_code}")
    except Exception as e:
        print_test("News - Invalid Parameter (negative)", "WARN",
                  f"Error: {str(e)}")
    
    # Test invalid parameter (zero)
    # FastAPI Query validation returns 422 Unprocessable Entity
    try:
        response = requests.get(f"{BASE_URL}/news", params={"days_back": 0}, timeout=10)
        if response.status_code == 422:
            print_test("News - Invalid Parameter (zero)", "PASS",
                      "Correctly rejected zero days_back (422)")
        elif response.status_code == 400:
            print_test("News - Invalid Parameter (zero)", "PASS",
                      "Correctly rejected zero days_back (400)")
        elif response.status_code == 200:
            print_test("News - Invalid Parameter (zero)", "FAIL",
                      "Should reject zero days_back (got 200)")
            all_passed = False
        else:
            print_test("News - Invalid Parameter (zero)", "WARN",
                      f"Unexpected status code: {response.status_code}")
    except Exception as e:
        print_test("News - Invalid Parameter (zero)", "WARN",
                  f"Error: {str(e)}")
    
    return all_passed


def test_storage_info():
    """Test GET /storage/info endpoint"""
    print_header("Test 5: Storage Info (GET /storage/info)")
    
    try:
        response = requests.get(f"{BASE_URL}/storage/info", timeout=10)
        
        # Check status code
        if response.status_code != 200:
            print_test("Storage Info - Status Code", "FAIL",
                      f"Expected 200, got {response.status_code}")
            return False
        
        print_test("Storage Info - Status Code", "PASS")
        
        # Check response structure
        data = response.json()
        required_keys = ["headlines", "predictions", "embeddings", "features"]
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print_test("Storage Info - Response Structure", "FAIL",
                      f"Missing keys: {missing_keys}")
            return False
        
        print_test("Storage Info - Response Structure", "PASS")
        
        # Check each storage section
        for section in required_keys:
            section_data = data[section]
            
            if "directory" not in section_data and "file" not in section_data:
                print_test(f"Storage Info - {section} Structure", "WARN",
                          "Missing directory/file key")
            else:
                print_test(f"Storage Info - {section} Structure", "PASS")
            
            if "count" in section_data:
                count = section_data["count"]
                print_test(f"Storage Info - {section} Count", "PASS",
                          f"{count} items")
            
            if section == "headlines" and "exists" in section_data:
                exists = section_data["exists"]
                status = "PASS" if exists else "WARN"
                print_test(f"Storage Info - Headlines File", status,
                          "Exists" if exists else "Not found")
        
        return True
        
    except requests.exceptions.Timeout:
        print_test("Storage Info", "FAIL", "Request timeout")
        return False
    except Exception as e:
        print_test("Storage Info", "FAIL", f"Unexpected error: {str(e)}")
        return False


def test_error_handling():
    """Test error handling for invalid endpoints"""
    print_header("Test 6: Error Handling")
    
    # Test non-existent endpoint
    try:
        response = requests.get(f"{BASE_URL}/nonexistent", timeout=5)
        if response.status_code == 404:
            print_test("Error Handling - 404", "PASS", "Correctly returns 404")
        else:
            print_test("Error Handling - 404", "WARN",
                      f"Expected 404, got {response.status_code}")
    except Exception as e:
        print_test("Error Handling - 404", "WARN", f"Error: {str(e)}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print_header("API Test Suite - Hybrid ESN-Ridge Prediction API")
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    if not check_server_running():
        print("\n❌ ERROR: API server is not running!")
        print(f"Please start the server first:")
        print(f"  cd application/backend")
        print(f"  python main.py")
        return False
    
    print("\n✓ Server is running")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Models Info", test_models_info),
        ("Predictions", test_predict),
        ("News Headlines", test_news),
        ("Storage Info", test_storage_info),
        ("Error Handling", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print_test(test_name, "FAIL", f"Test crashed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_header("Test Summary")
    
    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["warnings"])
    
    print(f"\nTotal Tests: {total}")
    print(f"✓ Passed: {len(test_results['passed'])}")
    print(f"✗ Failed: {len(test_results['failed'])}")
    print(f"⚠ Warnings: {len(test_results['warnings'])}")
    
    if test_results["failed"]:
        print("\nFailed Tests:")
        for test in test_results["failed"]:
            print(f"  ✗ {test}")
    
    if test_results["warnings"]:
        print("\nWarnings:")
        for test in test_results["warnings"]:
            print(f"  ⚠ {test}")
    
    success_rate = (len(test_results["passed"]) / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if len(test_results["failed"]) == 0:
        print("\n✅ All critical tests passed!")
        return True
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return False


def main():
    """Main entry point"""
    global BASE_URL
    
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--test", help="Run specific test (test_health_check, test_models_info, etc.)")
    parser.add_argument("--url", default=BASE_URL, help=f"API base URL (default: {BASE_URL})")
    
    args = parser.parse_args()
    
    BASE_URL = args.url
    
    if args.test:
        # Run specific test
        test_func = globals().get(args.test)
        if test_func and callable(test_func):
            print(f"Running specific test: {args.test}")
            test_func()
        else:
            print(f"Test '{args.test}' not found")
            print("Available tests:")
            for name in dir():
                if name.startswith("test_") and callable(globals()[name]):
                    print(f"  - {name}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

