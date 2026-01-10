"""
Setup Verification Script
Run this after installing requirements.txt to verify your environment
"""

import sys

def verify_setup():
    print("=" * 60)
    print("CREDIT RISK ML PROJECT - SETUP VERIFICATION")
    print("=" * 60)

    # Check Python version
    print(f"\n1. Python Version: {sys.version}")
    major, minor = sys.version_info[:2]

    if major == 3 and minor >= 8:
        print("   ✓ Python version is compatible (3.8+)")
    else:
        print("   ✗ Python 3.8+ required. Please upgrade Python.")
        return False

    # Check required packages
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'xgboost': 'Gradient boosting',
        'lightgbm': 'Gradient boosting',
        'imblearn': 'Imbalanced data handling',
        'shap': 'Model explainability',
        'mlflow': 'Experiment tracking',
        'fastapi': 'API development',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical visualization',
        'jupyter': 'Jupyter notebooks'
    }

    print("\n2. Checking Required Packages:")
    all_installed = True

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {package:15s} - {description}")
        except ImportError:
            print(f"   ✗ {package:15s} - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\n   Run: pip install -r requirements.txt")
        return False

    # Check directory structure
    import os
    print("\n3. Checking Directory Structure:")

    required_dirs = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'api',
        'models'
    ]

    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✓ {directory}/")
        else:
            print(f"   ✗ {directory}/ - Creating...")
            os.makedirs(directory, exist_ok=True)

    # Check for dataset
    print("\n4. Checking for Dataset:")
    data_dir = 'data/raw'

    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        if csv_files:
            print(f"   ✓ Found {len(csv_files)} CSV file(s):")
            for f in csv_files:
                file_size = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
                print(f"      - {f} ({file_size:.2f} MB)")
        else:
            print("   ⚠ No dataset found in data/raw/")
            print("      Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data")
            print("      Place CSV file in: data/raw/")

    # Check configuration
    print("\n5. Checking Configuration:")
    if os.path.exists('config.yaml'):
        print("   ✓ config.yaml exists")
    else:
        print("   ✗ config.yaml missing")
        return False

    # Version summary
    print("\n6. Package Versions:")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost as xgb

        print(f"   Pandas:       {pd.__version__}")
        print(f"   NumPy:        {np.__version__}")
        print(f"   Scikit-learn: {sklearn.__version__}")
        print(f"   XGBoost:      {xgb.__version__}")
    except Exception as e:
        print(f"   ⚠ Could not retrieve versions: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)

    if all_installed:
        print("\n✓ All checks passed! You're ready to start.")
        print("\nNext Steps:")
        print("1. Download dataset to data/raw/ (if not done)")
        print("2. Run: jupyter notebook")
        print("3. Open: notebooks/01_eda.ipynb")
        print("4. Start exploring your data!")
        return True
    else:
        print("\n✗ Some checks failed. Please review above.")
        return False

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
