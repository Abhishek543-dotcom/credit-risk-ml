# Complete Credit Risk ML Documentation
## A Beginner's Guide to Production Machine Learning

**Table of Contents**
- [1. Introduction](#1-introduction)
- [2. Business Problem](#2-business-problem)
- [3. Machine Learning Fundamentals](#3-machine-learning-fundamentals)
- [4. Credit Risk Concepts](#4-credit-risk-concepts)
- [5. Project Architecture](#5-project-architecture)
- [6. Data Pipeline](#6-data-pipeline)
- [7. Model Training](#7-model-training)
- [8. Model Deployment](#8-model-deployment)
- [9. Monitoring & Maintenance](#9-monitoring--maintenance)
- [10. Complete Workflow](#10-complete-workflow)

---

## 1. Introduction

### What is This Project?

This project is an **end-to-end machine learning system** that predicts whether a loan applicant will default (fail to repay) on their loan. It's a complete production-ready system that:

1. **Takes input**: Applicant information (age, income, debt ratio, etc.)
2. **Processes data**: Cleans and transforms the data
3. **Makes predictions**: Uses ML model to predict default probability
4. **Provides recommendations**: Approve or reject the loan
5. **Monitors performance**: Tracks predictions and model quality
6. **Auto-retrains**: Updates the model when performance degrades

### Why This Project Matters

Banks and financial institutions use systems like this to:
- **Reduce risk**: Avoid lending to people who won't repay
- **Increase profits**: Approve good borrowers who will repay
- **Meet regulations**: Provide explainable decisions
- **Automate decisions**: Process thousands of applications daily

---

## 2. Business Problem

### The Challenge

**Problem**: A bank receives 10,000 loan applications per month. Manually reviewing each application:
- Takes too long (5-7 days per application)
- Is expensive ($50 per review)
- Is inconsistent (different reviewers make different decisions)
- Misses patterns that indicate risk

**Impact of Bad Decisions**:
- **False Negative** (reject good borrower): Lost revenue (~$5,000 per loan)
- **False Positive** (approve bad borrower): Lost principal + interest (~$30,000 per default)

### The Solution

Build an ML system that:
1. **Predicts default probability** in milliseconds
2. **Processes 1000s of applications** per day
3. **Makes consistent decisions** based on data
4. **Learns from new data** continuously

### Success Metrics

- **ROC-AUC > 0.75**: Model can distinguish good from bad borrowers
- **Precision > 0.70**: 70% of approved loans won't default
- **Recall > 0.60**: Catch 60% of potential defaults
- **Response time < 200ms**: Fast enough for real-time decisions

---

## 3. Machine Learning Fundamentals

### What is Machine Learning?

**Machine Learning** is teaching computers to learn from data instead of explicit programming.

**Traditional Programming**:
```
Input → Rules (if/else) → Output
```

**Machine Learning**:
```
Input + Expected Output → Learning Algorithm → Model
Model + New Input → Prediction
```

### Types of Machine Learning

#### 1. Supervised Learning (What We Use)
- **Definition**: Learn from labeled examples
- **Example**: Given past loans with outcomes (default/no default), predict future loans
- **Our case**: Binary classification (default = 1, no default = 0)

#### 2. Unsupervised Learning
- **Definition**: Find patterns without labels
- **Example**: Group customers into segments
- **Not used here**: We have labels (default/no default)

#### 3. Reinforcement Learning
- **Definition**: Learn by trial and error
- **Example**: Game-playing AI
- **Not used here**: We have historical data

### Key ML Concepts

#### Features (Input Variables)
Variables used to make predictions:
- **age**: Applicant's age in years
- **MonthlyIncome**: Monthly income in dollars
- **DebtRatio**: Total debt divided by income
- **RevolvingUtilizationOfUnsecuredLines**: Credit card usage (0-1)

#### Target (Output Variable)
What we want to predict:
- **SeriousDlqin2yrs**: Did the borrower default? (0 = No, 1 = Yes)

#### Training vs Testing
- **Training Data (80%)**: Used to teach the model
- **Test Data (20%)**: Used to evaluate how well model learned

#### Overfitting vs Underfitting

**Underfitting** (Too Simple):
```
Model: Everyone with income < $3000 will default
Problem: Misses complex patterns
```

**Good Fit**:
```
Model: Considers income, debt ratio, age, credit usage together
Result: Captures real patterns
```

**Overfitting** (Too Complex):
```
Model: Memorizes training data exactly
Problem: Doesn't work on new data
```

---

## 4. Credit Risk Concepts

### What is Credit Risk?

**Credit Risk**: The probability that a borrower will fail to repay a loan.

### Risk Categories

```
Low Risk (0-30% default probability):
- Approve with standard rate
- Examples: High income, low debt, stable job

Medium Risk (30-50% default probability):
- Approve with higher interest rate
- Examples: Average income, moderate debt

High Risk (50-70% default probability):
- Approve with collateral or co-signer
- Examples: Low income, high debt

Very High Risk (70-100% default probability):
- Reject application
- Examples: Very high debt, previous defaults
```

### Important Features

#### 1. Debt Ratio
```
DebtRatio = Total Monthly Debt / Monthly Income

Examples:
- Income: $5000, Debt: $1500 → Ratio: 0.3 (Good)
- Income: $3000, Debt: $2700 → Ratio: 0.9 (Bad)
```

#### 2. Credit Utilization
```
Utilization = Credit Card Balance / Credit Card Limit

Examples:
- Limit: $10000, Balance: $2000 → 0.2 (Good)
- Limit: $5000, Balance: $4800 → 0.96 (Bad)
```

#### 3. Age
- Younger borrowers: Higher risk (less stable)
- Older borrowers: Lower risk (more established)
- Sweet spot: 35-55 years

### Class Imbalance Problem

**Problem**: In real data:
- 95% of loans DON'T default (class 0)
- 5% of loans DO default (class 1)

**Why This Matters**:
If model predicts "no default" for everyone → 95% accuracy!
But it's useless because it never catches defaults.

**Solutions We Use**:
1. **SMOTE** (Synthetic Minority Over-sampling): Creates synthetic examples of defaults
2. **Class Weights**: Penalizes model more for missing defaults
3. **Evaluation Metrics**: Use ROC-AUC instead of accuracy

---

## 5. Project Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER/CLIENT                           │
│              (Web App, Mobile App, API Client)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER                           │
│                   (Port 8000)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  /health      → Health check                        │   │
│  │  /predict     → Single prediction                   │   │
│  │  /predict/batch → Batch predictions                 │   │
│  │  /model/info  → Model information                   │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION PIPELINE                         │
│                  (src/inference.py)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Validate Input                                  │   │
│  │  2. Feature Engineering                             │   │
│  │  3. Scale Features                                  │   │
│  │  4. Model Prediction                                │   │
│  │  5. Risk Classification                             │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML MODEL (XGBoost)                        │
│               (models/production_model.pkl)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Input: Scaled features [age, income, debt, ...]   │   │
│  │  Output: Probability [0.2341]                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────┐
        │      MONITORING SYSTEM             │
        │   (src/monitoring.py)              │
        │  - Log predictions                 │
        │  - Track performance               │
        │  - Detect drift                    │
        └────────────────────────────────────┘

        ┌────────────────────────────────────┐
        │    RETRAINING PIPELINE             │
        │    (src/retrain.py)                │
        │  - Load new data                   │
        │  - Train new model                 │
        │  - Deploy if better                │
        └────────────────────────────────────┘
```

### Component Breakdown

#### 1. API Layer (api/main.py)
**Purpose**: Handle HTTP requests from clients

**Responsibilities**:
- Validate incoming data
- Call prediction pipeline
- Format responses
- Handle errors
- Log requests

**Example Flow**:
```
Client sends POST request
  ↓
FastAPI receives JSON
  ↓
Pydantic validates input
  ↓
Call predictor.predict()
  ↓
Return JSON response
```

#### 2. Prediction Pipeline (src/inference.py)
**Purpose**: Transform raw input into predictions

**Steps**:
```python
1. Load Input
   {"age": 45, "MonthlyIncome": 5000, ...}

2. Feature Engineering
   debt_to_income_ratio = DebtRatio * MonthlyIncome
   credit_utilization_bucket = bucket(RevolvingUtilization)

3. Handle Missing Values
   Fill with median/mode

4. Scale Features
   Standardize to mean=0, std=1

5. Model Prediction
   XGBoost returns probability [0.2341]

6. Risk Classification
   if prob < 0.3: "Low Risk"
   elif prob < 0.5: "Medium Risk"
   elif prob < 0.7: "High Risk"
   else: "Very High Risk"

7. Return Result
   {
     "prediction": 0,
     "probability": 0.2341,
     "risk_level": "Low",
     "recommendation": "Approve"
   }
```

#### 3. ML Model (XGBoost)
**What is XGBoost?**

XGBoost = eXtreme Gradient Boosting

**Simple Explanation**:
Imagine you're trying to guess someone's salary:
- Tree 1: "If age > 30, guess $50k, else $30k"
- Tree 2: "Actually, if has degree, add $20k"
- Tree 3: "But if debt > 50%, subtract $10k"
- Final prediction = Sum of all tree predictions

**Why XGBoost is Good for Credit Risk**:
1. **Handles non-linear patterns**: Income + debt interact
2. **Feature importance**: Shows which features matter
3. **Fast predictions**: Can process 1000s per second
4. **Robust**: Handles missing values and outliers

**How XGBoost Works**:
```
Step 1: Build first tree (simple guess)
  Prediction: 0.5 (50% default chance for everyone)

Step 2: Calculate errors
  True: 0, Predicted: 0.5 → Error: -0.5
  True: 1, Predicted: 0.5 → Error: +0.5

Step 3: Build tree to predict errors
  "If DebtRatio > 0.7, adjust by +0.3"

Step 4: Add correction
  New Prediction = 0.5 + 0.3 = 0.8

Repeat 100 times → Final Model
```

#### 4. Monitoring System (src/monitoring.py)
**Purpose**: Track model health in production

**What It Monitors**:

1. **Prediction Distribution**
   ```
   Daily Approval Rate:
   Monday: 68% approved, 32% rejected
   Tuesday: 70% approved, 30% rejected
   ...
   Sudden change to 50%? → Alert!
   ```

2. **Data Drift**
   ```
   Training Data:
   - Average age: 45
   - Average income: $5000

   Production Data:
   - Average age: 52 (drift!)
   - Average income: $4500 (drift!)

   → Model may perform poorly on new data
   ```

3. **Performance Metrics**
   ```
   Training ROC-AUC: 0.82
   Production ROC-AUC: 0.75 (degraded!)

   → Time to retrain model
   ```

#### 5. Retraining Pipeline (src/retrain.py)
**Purpose**: Keep model fresh with new data

**Retraining Workflow**:
```
1. Load New Data
   - Last 6 months of loan outcomes
   - Confirmed defaults and repayments

2. Preprocess
   - Same steps as training
   - Feature engineering
   - Handle missing values
   - Scale features

3. Train New Model
   - Same algorithm (XGBoost)
   - Same hyperparameters
   - New data patterns

4. Evaluate
   - Test on hold-out set
   - Calculate ROC-AUC, precision, recall

5. Compare with Current Model
   - New ROC-AUC: 0.84
   - Current ROC-AUC: 0.82
   - Improvement: +0.02 (2%)

6. Deploy if Better
   - If improvement > 1%: Deploy
   - Backup old model
   - Save new model
   - Update production
```

---

## 6. Data Pipeline

### Data Flow

```
┌──────────────────────┐
│   RAW DATA           │
│   (CSV file)         │
│  - 150,000 rows      │
│  - 11 columns        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  1. DATA LOADING     │
│  pandas.read_csv()   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. EXPLORATORY      │
│     DATA ANALYSIS    │
│  - Check shape       │
│  - Check nulls       │
│  - Plot distributions│
│  - Correlation       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. DATA CLEANING    │
│  - Handle missing    │
│  - Remove outliers   │
│  - Fix data types    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. FEATURE          │
│     ENGINEERING      │
│  - Create new vars   │
│  - Transform         │
│  - Encode categories │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  5. TRAIN-TEST SPLIT │
│  - Training: 80%     │
│  - Testing: 20%      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  6. SCALING          │
│  StandardScaler      │
│  mean=0, std=1       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  7. HANDLE IMBALANCE │
│  SMOTE: Oversample   │
│  minority class      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  READY FOR TRAINING  │
│  X_train, y_train    │
│  X_test, y_test      │
└──────────────────────┘
```

### Step-by-Step Explanation

#### Step 1: Data Loading
```python
# Load CSV file
df = pd.read_csv('data/raw/credit_data.csv')

# Check basic info
print(df.shape)  # (150000, 11)
print(df.columns)
# ['SeriousDlqin2yrs', 'age', 'MonthlyIncome', 'DebtRatio', ...]
```

#### Step 2: Exploratory Data Analysis (EDA)

**Purpose**: Understand the data before modeling

**Key Questions**:
1. How many records? (150,000 loans)
2. How many features? (10 features + 1 target)
3. Missing values? (Yes, 20% in MonthlyIncome)
4. Target distribution? (93% class 0, 7% class 1 - imbalanced!)
5. Feature distributions? (Income is right-skewed, age is normal)

**Example Analysis**:
```python
# Target distribution
print(df['SeriousDlqin2yrs'].value_counts())
# 0 (no default): 139,974 (93.3%)
# 1 (default):     10,026 (6.7%)
# → Imbalanced!

# Missing values
print(df.isnull().sum())
# MonthlyIncome: 29,731 (19.8%)
# → Need to handle!

# Age distribution
df['age'].hist()
# Bell curve, centered at 45
```

#### Step 3: Data Cleaning

**Handle Missing Values**:
```python
# Option 1: Fill with median (for numerical)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

# Option 2: Fill with mode (for categorical)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Option 3: Drop rows (if < 5% missing)
df.dropna(inplace=True)
```

**Handle Outliers**:
```python
# Identify outliers (values beyond 3 standard deviations)
z_scores = (df['MonthlyIncome'] - df['MonthlyIncome'].mean()) / df['MonthlyIncome'].std()
outliers = abs(z_scores) > 3

# Option 1: Cap at 99th percentile
upper_limit = df['MonthlyIncome'].quantile(0.99)
df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=upper_limit)

# Option 2: Remove outliers
df = df[~outliers]
```

#### Step 4: Feature Engineering

**What is Feature Engineering?**
Creating new features from existing ones to help the model learn better.

**Examples**:

1. **Debt-to-Income Ratio**
```python
# Why: DebtRatio alone doesn't tell full story
# Someone with 0.5 debt ratio but $10k income is better than
# someone with 0.5 debt ratio but $2k income

df['debt_to_income'] = df['DebtRatio'] * df['MonthlyIncome']

# Example:
# Person A: DebtRatio=0.5, Income=$10000 → debt_to_income=$5000
# Person B: DebtRatio=0.5, Income=$2000 → debt_to_income=$1000
# Model can now distinguish these better!
```

2. **Credit Utilization Buckets**
```python
# Why: Non-linear relationship
# 0-30%: Excellent
# 30-70%: Good
# 70-100%: Warning
# >100%: Danger!

df['credit_bucket'] = pd.cut(
    df['RevolvingUtilizationOfUnsecuredLines'],
    bins=[0, 0.3, 0.7, 1.0, float('inf')],
    labels=[0, 1, 2, 3]  # Encoded as numbers
)
```

3. **Age Groups**
```python
# Young (18-25): High risk
# Adult (26-50): Medium risk
# Senior (51+): Low risk

df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 50, 100],
    labels=['young', 'adult', 'senior']
)
```

#### Step 5: Train-Test Split

**Why Split?**
- **Training Set**: Teach the model
- **Test Set**: Evaluate how well it learned (unseen data)

```python
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# Split 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Keep same class distribution
)

# Result:
# X_train: 120,000 samples (80%)
# X_test:   30,000 samples (20%)
```

**Stratification**:
```
Without stratify:
Train: 95% class 0, 5% class 1
Test:  90% class 0, 10% class 1 (different!)

With stratify:
Train: 93.3% class 0, 6.7% class 1
Test:  93.3% class 0, 6.7% class 1 (same!)
```

#### Step 6: Feature Scaling

**Why Scale?**
Different features have different ranges:
- Age: 18-90 (small range)
- MonthlyIncome: 0-100,000 (large range)

Models are sensitive to scale. Scaling makes all features comparable.

**StandardScaler (Z-score normalization)**:
```python
from sklearn.preprocessing import StandardScaler

# Formula: (value - mean) / std
scaler = StandardScaler()

# Fit on training data ONLY
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Example transformation:
# Original: age=45, income=5000
# Scaled:   age=0.0, income=0.2
```

**Before vs After**:
```
Before Scaling:
age:            [18, 25, 45, 60, 75]
MonthlyIncome:  [2000, 3500, 5000, 8000, 12000]
→ Income dominates because larger numbers

After Scaling:
age:            [-1.5, -0.8, 0.0, 0.8, 1.5]
MonthlyIncome:  [-1.2, -0.5, 0.0, 0.9, 1.8]
→ Both features equally important
```

#### Step 7: Handle Class Imbalance

**The Problem**:
```
Training Data:
Class 0 (no default): 111,600 samples (93%)
Class 1 (default):      8,400 samples (7%)

Model learns:
"Just predict 0 for everyone" → 93% accuracy!
But useless for catching defaults.
```

**Solution: SMOTE** (Synthetic Minority Over-sampling Technique)

**How SMOTE Works**:
```
1. Find minority class samples (defaults)
2. For each sample:
   - Find k nearest neighbors (similar defaults)
   - Create synthetic samples between them

Example:
Sample A: age=25, income=2000, debt=0.8
Sample B: age=28, income=2200, debt=0.85

Synthetic: age=26.5, income=2100, debt=0.825
(Point between A and B)

3. Add synthetic samples until balanced
```

**Code**:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)
# sampling_strategy=0.5 means make minority 50% of majority

X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Before SMOTE:
# Class 0: 111,600 samples
# Class 1:   8,400 samples

# After SMOTE:
# Class 0: 111,600 samples (unchanged)
# Class 1:  55,800 samples (created ~47,400 synthetic)
```

---

## 7. Model Training

### Training Pipeline

```
┌────────────────────────────────────────┐
│  1. CHOOSE ALGORITHM                   │
│  - Logistic Regression (baseline)     │
│  - Random Forest (ensemble)            │
│  - XGBoost (final choice)              │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  2. SET HYPERPARAMETERS                │
│  - n_estimators: 100                   │
│  - max_depth: 6                        │
│  - learning_rate: 0.1                  │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  3. TRAIN MODEL                        │
│  model.fit(X_train, y_train)           │
│  - Builds 100 decision trees           │
│  - Each tree learns from errors        │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  4. MAKE PREDICTIONS                   │
│  y_pred = model.predict(X_test)        │
│  y_proba = model.predict_proba(X_test) │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  5. EVALUATE PERFORMANCE               │
│  - ROC-AUC Score                       │
│  - Precision, Recall, F1               │
│  - Confusion Matrix                    │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  6. TUNE HYPERPARAMETERS               │
│  GridSearchCV / RandomizedSearchCV     │
│  - Try different combinations          │
│  - Find best parameters                │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  7. SAVE BEST MODEL                    │
│  joblib.dump(model, 'best_model.pkl')  │
└────────────────────────────────────────┘
```

### Algorithm Comparison

#### 1. Logistic Regression (Baseline)

**What It Is**: Linear model for classification

**How It Works**:
```
Equation: P(default) = 1 / (1 + e^-(b0 + b1*age + b2*income + ...))

Example:
P(default) = 1 / (1 + e^-(0.5 - 0.01*age + 0.0001*income + 2*debt))

For age=45, income=5000, debt=0.3:
P(default) = 1 / (1 + e^-(0.5 - 0.45 + 0.5 + 0.6))
           = 1 / (1 + e^-1.15)
           = 0.76 (76% chance of default)
```

**Pros**:
- Fast to train
- Easy to interpret
- Good baseline

**Cons**:
- Assumes linear relationships
- Can't capture complex patterns
- Performance: ROC-AUC ~ 0.68

#### 2. Random Forest

**What It Is**: Ensemble of decision trees

**How It Works**:
```
1. Build 100 decision trees
2. Each tree uses random subset of features
3. Each tree votes on prediction
4. Final prediction = majority vote

Tree 1: "Based on age and income → No default"
Tree 2: "Based on debt and credit → Default"
Tree 3: "Based on age and debt → No default"
...
Tree 100: "No default"

Final: 70 trees say "No default" → Predict "No default"
```

**Pros**:
- Handles non-linear patterns
- Less overfitting than single tree
- Feature importance

**Cons**:
- Slower than logistic regression
- Can overfit if too many trees
- Performance: ROC-AUC ~ 0.75

#### 3. XGBoost (Our Choice)

**What It Is**: Gradient boosting ensemble

**How It Works**:
```
Step 1: Build simple tree (baseline)
  Everyone gets predicted: 0.5

Step 2: Calculate errors
  True=0, Predicted=0.5 → Error=-0.5
  True=1, Predicted=0.5 → Error=+0.5

Step 3: Build tree to predict errors
  "If debt > 0.7, add +0.3 to prediction"

Step 4: Update predictions
  New = Old + (learning_rate × correction)
  New = 0.5 + (0.1 × 0.3) = 0.53

Repeat 100 times
```

**Pros**:
- Best performance (ROC-AUC ~ 0.82)
- Handles complex patterns
- Built-in regularization
- Fast predictions

**Cons**:
- More hyperparameters to tune
- Can overfit if not careful
- Harder to interpret

### Training Code

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. Initialize model
model = XGBClassifier(
    n_estimators=100,        # Number of trees
    max_depth=6,             # Maximum tree depth
    learning_rate=0.1,       # Shrinkage rate
    min_child_weight=1,      # Minimum samples per leaf
    subsample=0.8,           # Row sampling (80%)
    colsample_bytree=0.8,    # Column sampling (80%)
    scale_pos_weight=10,     # Weight for minority class
    eval_metric='auc',       # Evaluation metric
    random_state=42          # Reproducibility
)

# 2. Train model
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=10
)

# 3. Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 4. Evaluate
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")  # 0.8234

# 5. Detailed metrics
print(classification_report(y_test, y_pred))
```

### Hyperparameter Tuning

**What Are Hyperparameters?**
Settings you configure before training (not learned from data)

**Key Hyperparameters**:

1. **n_estimators** (Number of trees)
   ```
   Too few (10): Underfitting, poor performance
   Just right (100): Good balance
   Too many (1000): Overfitting, slow training
   ```

2. **max_depth** (Tree depth)
   ```
   Shallow (3): Simple patterns, underfitting
   Medium (6): Good balance
   Deep (15): Complex patterns, overfitting
   ```

3. **learning_rate** (Step size)
   ```
   High (0.3): Fast learning, may overshoot
   Medium (0.1): Good balance
   Low (0.01): Slow learning, need more trees
   ```

4. **scale_pos_weight** (Class imbalance)
   ```
   Formula: (# negative samples) / (# positive samples)
   Example: 111,600 / 8,400 = 13.3

   Effect: Penalizes model more for missing defaults
   ```

**Grid Search** (Try all combinations):
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc'
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
# {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
```

### Model Evaluation

#### Metrics Explained

**1. Confusion Matrix**
```
                Predicted
              No Default  Default
Actual  No    27,930      1,050    (True Neg + False Pos)
Default        420        600      (False Neg + True Pos)

True Negative (TN): 27,930 - Correctly predicted no default
False Positive (FP): 1,050 - Wrongly predicted default (Type I error)
False Negative (FN): 420 - Wrongly predicted no default (Type II error)
True Positive (TP): 600 - Correctly predicted default
```

**2. Precision**
```
Precision = TP / (TP + FP)
          = 600 / (600 + 1,050)
          = 0.36 (36%)

Meaning: Of all loans we rejected, 36% would actually default.
Business: 64% of rejections are lost revenue!
```

**3. Recall (Sensitivity)**
```
Recall = TP / (TP + FN)
       = 600 / (600 + 420)
       = 0.59 (59%)

Meaning: We catch 59% of actual defaults.
Business: 41% of defaults slip through!
```

**4. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.36 × 0.59) / (0.36 + 0.59)
   = 0.45

Meaning: Harmonic mean of precision and recall.
```

**5. ROC-AUC**
```
ROC Curve: True Positive Rate vs False Positive Rate
AUC: Area Under the Curve

AUC = 0.5: Random guessing (coin flip)
AUC = 0.7: Acceptable
AUC = 0.8: Good
AUC = 0.9: Excellent

Our model: AUC = 0.82 (Good!)

Meaning: 82% chance model ranks a random default higher
than a random non-default.
```

#### Choosing the Right Threshold

**Default threshold: 0.5**
```
If P(default) ≥ 0.5 → Predict default (reject loan)
If P(default) < 0.5 → Predict no default (approve loan)
```

**Business-Driven Thresholds**:

**Conservative Bank** (minimize defaults):
```
Threshold: 0.3
If P(default) ≥ 0.3 → Reject

Result:
- Catch more defaults (high recall)
- Reject more good borrowers (low precision)
- Lost revenue from rejections
```

**Aggressive Bank** (maximize approvals):
```
Threshold: 0.7
If P(default) ≥ 0.7 → Reject

Result:
- Approve more borrowers
- More defaults slip through
- Lost money from defaults
```

**Optimal Threshold** (maximize profit):
```
Cost of False Positive: $5,000 (lost revenue)
Cost of False Negative: $30,000 (default loss)

Find threshold that minimizes:
Total Cost = (FP × $5,000) + (FN × $30,000)

Our optimal: 0.4
```

---

## 8. Model Deployment

### Deployment Architecture

```
Development → Staging → Production

┌──────────────────────────────────────────────────────┐
│                 DEVELOPMENT                          │
│  - Train models locally                              │
│  - Experiment with features                          │
│  - Test on local data                                │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│                 STAGING                               │
│  - Deploy to test environment                        │
│  - Run integration tests                             │
│  - Performance testing                               │
│  - Security testing                                  │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│                 PRODUCTION                            │
│  - Deploy to live servers                            │
│  - Serve real traffic                                │
│  - Monitor performance                               │
│  - Auto-scaling                                      │
└──────────────────────────────────────────────────────┘
```

### Deployment Steps

#### Step 1: Prepare Model for Production

```python
# 1. Train final model on ALL data
X_all = np.concatenate([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

model.fit(X_all, y_all)

# 2. Save model
import joblib
joblib.dump(model, 'models/production_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# 3. Save metadata
metadata = {
    'model_type': 'XGBClassifier',
    'features': list(X.columns),
    'training_date': datetime.now().isoformat(),
    'roc_auc': 0.82,
    'version': '1.0.0'
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f)
```

#### Step 2: Create Prediction API

**Why API?**
- Standardized interface
- Language-agnostic (any client can use)
- Scalable (handle multiple requests)
- Secure (authentication, rate limiting)

**FastAPI Implementation**:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model once at startup
model = joblib.load('models/production_model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI()

# Define input schema
class LoanApplication(BaseModel):
    age: float
    MonthlyIncome: float
    DebtRatio: float
    RevolvingUtilizationOfUnsecuredLines: float

# Prediction endpoint
@app.post("/predict")
def predict(application: LoanApplication):
    # 1. Convert to dict
    data = application.dict()

    # 2. Feature engineering
    data['debt_to_income'] = data['DebtRatio'] * data['MonthlyIncome']

    # 3. Create feature array
    features = [data['age'], data['MonthlyIncome'], ...]

    # 4. Scale
    features_scaled = scaler.transform([features])

    # 5. Predict
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = int(probability >= 0.5)

    # 6. Classify risk
    if probability < 0.3:
        risk = "Low"
    elif probability < 0.5:
        risk = "Medium"
    elif probability < 0.7:
        risk = "High"
    else:
        risk = "Very High"

    # 7. Return response
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk,
        "recommendation": "Reject" if prediction == 1 else "Approve"
    }
```

#### Step 3: Containerize with Docker

**What is Docker?**
Package your application with all dependencies into a container.

**Dockerfile**:
```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY api/ api/
COPY src/ src/
COPY models/ models/
COPY config.yaml .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**:
```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 credit-risk-api

# Access API
curl http://localhost:8000/health
```

#### Step 4: Deploy to Cloud

**Option 1: AWS EC2**
```bash
# 1. Launch EC2 instance
aws ec2 run-instances --image-id ami-xxxxx --instance-type t2.medium

# 2. SSH into instance
ssh -i key.pem ec2-user@ec2-xxx.amazonaws.com

# 3. Install Docker
sudo yum install docker -y
sudo service docker start

# 4. Pull and run container
docker pull your-repo/credit-risk-api:latest
docker run -d -p 80:8000 credit-risk-api
```

**Option 2: AWS Lambda (Serverless)**
```python
# Serverless framework
# No server management
# Pay per request
# Auto-scaling

# deploy
sls deploy
```

**Option 3: Kubernetes**
```yaml
# Deployment manifest
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-api
spec:
  replicas: 3  # 3 instances for high availability
  selector:
    matchLabels:
      app: credit-risk-api
  template:
    metadata:
      labels:
        app: credit-risk-api
    spec:
      containers:
      - name: api
        image: credit-risk-api:latest
        ports:
        - containerPort: 8000
```

#### Step 5: Set Up Load Balancer

**Why Load Balancer?**
Distribute traffic across multiple API instances.

```
                ┌──────────────┐
Client ────────►│ Load Balancer│
                └───────┬──────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │ API #1 │    │ API #2 │    │ API #3 │
    └────────┘    └────────┘    └────────┘

Benefits:
- High availability (if one fails, others handle)
- Load distribution (handle more requests)
- Zero-downtime deployment
```

---

## 9. Monitoring & Maintenance

### What to Monitor

#### 1. Model Performance

**Metrics to Track**:
```python
{
    "date": "2024-01-10",
    "predictions": 1500,
    "roc_auc": 0.81,        # vs training: 0.82
    "precision": 0.68,      # vs training: 0.70
    "recall": 0.61,         # vs training: 0.62
    "f1_score": 0.64,

    "degradation": {
        "roc_auc": -0.01,   # -1.2% (acceptable)
        "precision": -0.02, # -2.9% (warning!)
        "recall": -0.01     # -1.6% (acceptable)
    }
}
```

**Alert Conditions**:
```
if degradation > 5%:
    send_alert("Model performance degraded!")
    trigger_retraining()
```

#### 2. Data Drift

**What is Data Drift?**
When production data distribution differs from training data.

**Example**:
```
Training Data:
- Average age: 45 years
- Average income: $5,000
- Average debt ratio: 0.35

Production Data (Month 6):
- Average age: 52 years (drift!)
- Average income: $4,200 (drift!)
- Average debt ratio: 0.42 (drift!)

Why: Economic recession → older, poorer applicants
Effect: Model performs poorly (trained on different data)
```

**Detection**:
```python
# Calculate distribution shift
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(
    training_data['age'],
    production_data['age']
)

if p_value < 0.05:
    print("Significant drift detected in age!")
    # Trigger retraining
```

#### 3. Prediction Distribution

**Monitor**:
```python
daily_stats = {
    "date": "2024-01-10",
    "total_predictions": 1500,
    "approvals": 1050,        # 70%
    "rejections": 450,        # 30%

    "risk_distribution": {
        "low": 600,           # 40%
        "medium": 450,        # 30%
        "high": 300,          # 20%
        "very_high": 150      # 10%
    },

    "avg_probability": 0.34,
    "std_probability": 0.22
}
```

**Anomaly Detection**:
```
Normal: 70% approval rate
Sudden: 50% approval rate

Possible causes:
- Data drift
- Model degradation
- Business change (different applicants)
```

#### 4. API Performance

**Metrics**:
```python
{
    "latency": {
        "p50": 45,     # 50% of requests < 45ms
        "p95": 120,    # 95% of requests < 120ms
        "p99": 250,    # 99% of requests < 250ms
        "max": 850
    },

    "throughput": {
        "requests_per_second": 150,
        "requests_per_minute": 9000
    },

    "errors": {
        "4xx_rate": 0.5,   # Client errors: 0.5%
        "5xx_rate": 0.1    # Server errors: 0.1%
    },

    "availability": 99.95  # 99.95% uptime
}
```

**SLA (Service Level Agreement)**:
```
Guaranteed:
- 99.9% uptime (< 43 minutes downtime/month)
- p95 latency < 200ms
- Error rate < 1%
```

### Monitoring Dashboard

**Grafana Dashboard Example**:
```
┌────────────────────────────────────────────────────┐
│ Credit Risk API - Production Dashboard            │
├────────────────────────────────────────────────────┤
│                                                    │
│  Requests/sec: 125 ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁               │
│  Latency p95:  95ms ▁▁▂▂▃▃▄▄▃▃▂▂▁▁               │
│  Error Rate:   0.3% ▁▁▁▁▁▁▁▁▁▁▁▁▁▁               │
│                                                    │
│  Today's Predictions                               │
│  ├─ Total: 15,234                                 │
│  ├─ Approved: 10,664 (70%)                        │
│  └─ Rejected: 4,570 (30%)                         │
│                                                    │
│  Model Performance                                 │
│  ├─ ROC-AUC: 0.81 ⚠️ (down from 0.82)            │
│  ├─ Precision: 0.68                               │
│  └─ Recall: 0.61                                  │
│                                                    │
│  Data Drift Status                                 │
│  ├─ Age: ✅ No drift                             │
│  ├─ Income: ⚠️ Warning (5% shift)                │
│  └─ Debt Ratio: ❌ Drift detected (12% shift)    │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Automated Retraining

**Trigger Conditions**:
```python
def should_retrain():
    # 1. Check performance degradation
    if current_roc_auc < baseline_roc_auc - 0.05:
        return True, "Performance degraded > 5%"

    # 2. Check data drift
    if drift_detected():
        return True, "Data drift detected"

    # 3. Check time since last training
    days_since_training = (datetime.now() - last_train_date).days
    if days_since_training > 90:
        return True, "90 days since last training"

    # 4. Check amount of new data
    if new_data_count > 50000:
        return True, "Sufficient new data available"

    return False, "No retraining needed"
```

**Retraining Workflow**:
```
1. Trigger Detected
   ↓
2. Collect New Data
   - Last 6 months of loans
   - Confirmed outcomes
   ↓
3. Retrain Model
   - Same preprocessing
   - Same algorithm
   - New data
   ↓
4. Validate Model
   - Test on hold-out set
   - Compare with current model
   ↓
5. A/B Test (Optional)
   - Send 10% traffic to new model
   - Compare metrics
   ↓
6. Deploy if Better
   - Backup current model
   - Deploy new model
   - Monitor closely
   ↓
7. Rollback if Issues
   - Revert to previous model
   - Investigate problems
```

---

## 10. Complete Workflow

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│                    STEP 1: DATA COLLECTION              │
│  Business collects loan applications and outcomes       │
│  Store in database: 150,000 historical loans           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 2: EXPLORATORY ANALYSIS         │
│  Notebook: 01_eda.ipynb                                │
│  - Load data                                            │
│  - Check distributions                                  │
│  - Find missing values                                  │
│  - Analyze correlations                                 │
│  - Visualize patterns                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 3: FEATURE ENGINEERING          │
│  Notebook: 02_feature_engineering.ipynb                │
│  - Handle missing values                                │
│  - Create new features                                  │
│  - Transform variables                                  │
│  - Encode categories                                    │
│  - Save processed data                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 4: MODEL TRAINING               │
│  Notebook: 03_model_training.ipynb                     │
│  - Split train/test                                     │
│  - Scale features                                       │
│  - Handle imbalance (SMOTE)                             │
│  - Train models (Logistic, RF, XGBoost)                 │
│  - Compare performance                                  │
│  - Select best model                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 5: MODEL EVALUATION             │
│  Notebook: 04_model_evaluation.ipynb                   │
│  - Calculate metrics (ROC-AUC, precision, recall)       │
│  - Plot confusion matrix                                │
│  - Analyze errors                                       │
│  - Feature importance                                   │
│  - Business impact analysis                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 6: MODEL DEPLOYMENT             │
│  Script: src/inference.py + api/main.py                │
│  - Save best model                                      │
│  - Create prediction pipeline                           │
│  - Build REST API                                       │
│  - Containerize with Docker                             │
│  - Deploy to cloud                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 7: MONITORING                   │
│  Script: src/monitoring.py                             │
│  - Log all predictions                                  │
│  - Track performance metrics                            │
│  - Detect data drift                                    │
│  - Generate reports                                     │
│  - Alert on issues                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    STEP 8: RETRAINING                   │
│  Script: src/retrain.py                                │
│  - Collect new data                                     │
│  - Retrain model                                        │
│  - Validate performance                                 │
│  - Deploy if improved                                   │
│  - Backup old model                                     │
└─────────────────────────────────────────────────────────┘
                   │
                   │ (Loop back to Step 7)
                   └─────────────────────┐
                                         │
                                         ▼
                              (Continuous Improvement)
```

### Real-World Example

**Scenario**: A bank wants to automate loan approvals

**Day 1-7: Data Collection**
```
Bank exports historical data:
- 150,000 loans from last 5 years
- Features: age, income, debt, credit score, etc.
- Outcomes: default (yes/no)

Data Engineer saves to: data/raw/loans.csv
```

**Day 8-14: Exploration & Feature Engineering**
```
Data Scientist runs:
1. notebooks/01_eda.ipynb
   - Finds 20% missing in MonthlyIncome
   - Discovers class imbalance (7% defaults)
   - Identifies correlated features

2. notebooks/02_feature_engineering.ipynb
   - Fills missing with median
   - Creates debt_to_income ratio
   - Buckets credit utilization
   - Saves to: data/processed/train.csv
```

**Day 15-21: Model Training**
```
Data Scientist runs:
1. notebooks/03_model_training.ipynb
   - Trains 3 models
   - XGBoost wins: ROC-AUC = 0.82

2. notebooks/04_model_evaluation.ipynb
   - Precision: 70%
   - Recall: 62%
   - Business approves metrics

Saves: models/best_model.pkl
```

**Day 22-28: Deployment**
```
ML Engineer:
1. Creates API (api/main.py)
2. Builds Docker image
3. Deploys to AWS
4. Sets up load balancer
5. Configures monitoring

API is live: https://api.bank.com/predict
```

**Day 29+: Production**
```
Daily Operations:
- 1,500 loan applications/day
- API processes in < 100ms
- 70% approval rate
- Monitoring dashboard tracks health

Weekly:
- Review monitoring reports
- Check for drift
- Validate performance

Monthly:
- Retrain with new data (if needed)
- Deploy updated model
- A/B test performance
```

**Results After 6 Months**:
```
Business Impact:
- Processing time: 5 days → 1 second (99.99% faster)
- Cost per review: $50 → $0.01 (99.98% cheaper)
- Consistency: Variable → 100%
- Caught defaults: +35% (saved $5M)
- Approved good loans: +20% (earned $10M)

Total ROI: $15M saved/earned in 6 months
```

### Success Metrics

**Technical Metrics**:
- ✅ ROC-AUC: 0.82 (target: > 0.75)
- ✅ Precision: 0.70 (target: > 0.65)
- ✅ Recall: 0.62 (target: > 0.60)
- ✅ API Latency: 95ms (target: < 200ms)
- ✅ Uptime: 99.95% (target: > 99.9%)

**Business Metrics**:
- ✅ Default rate: 6.5% (down from 7.2%)
- ✅ Approval rate: 70% (up from 65%)
- ✅ Processing time: 1s (down from 5 days)
- ✅ Cost per application: $0.01 (down from $50)
- ✅ Annual savings: $15M

---

## Conclusion

### What You've Learned

**Machine Learning Fundamentals**:
- Supervised learning for classification
- Feature engineering and preprocessing
- Model training and evaluation
- Hyperparameter tuning
- Handling imbalanced data

**Credit Risk Domain**:
- Understanding default probability
- Risk classification
- Business metrics (ROC-AUC, precision, recall)
- Cost-benefit analysis

**Production ML (MLOps)**:
- Building prediction pipelines
- Creating REST APIs
- Docker containerization
- Cloud deployment
- Monitoring and alerting
- Automated retraining

**Software Engineering**:
- API design (FastAPI)
- Testing and validation
- Documentation
- Version control
- CI/CD pipelines

### Next Steps

**Beginner** (You are here!):
- ✅ Understand the complete workflow
- ✅ Run the notebooks
- ✅ Deploy the API locally
- ✅ Make test predictions

**Intermediate**:
- [ ] Add SHAP explanations for interpretability
- [ ] Integrate MLflow for experiment tracking
- [ ] Set up Grafana dashboards
- [ ] Implement A/B testing framework
- [ ] Add API authentication

**Advanced**:
- [ ] Build real-time streaming pipeline
- [ ] Implement online learning
- [ ] Create custom loss functions
- [ ] Deploy on Kubernetes
- [ ] Build AutoML pipeline

### Resources for Further Learning

**Books**:
- "Hands-On Machine Learning" by Aurélien Géron
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Machine Learning Engineering" by Andriy Burkov

**Courses**:
- fast.ai - Practical Deep Learning
- Andrew Ng's Machine Learning (Coursera)
- Full Stack Deep Learning

**Tools to Explore**:
- MLflow: Experiment tracking
- Weights & Biases: Model monitoring
- Seldon Core: ML deployment
- Kubeflow: ML on Kubernetes

---

**Congratulations!** 🎉

You now understand how to build, deploy, and maintain a production machine learning system from scratch!

This documentation covers the complete journey from raw data to a live API serving predictions at scale.

**Author**: Abhishek Tiwari
**Last Updated**: 2024-01-10
**Version**: 1.0.0
