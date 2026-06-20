import os
import sys
import pandas as pd
import numpy as np
import pytest

# Fix import path for local modules when running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_feature import build_features
from src.serving.inference import predict

def test_preprocess_data():
    # Create mock data mimicking Telco Churn structure
    mock_data = pd.DataFrame({
        'customerID': ['1234-ABCD', '5678-EFGH'],
        'gender': ['Male', 'Female'],
        'MonthlyCharges': [50.0, '60.0'],
        'tenure': [10, 5],
        'TotalCharges': ['500.0', ' '],  # space tests coercion
        'Churn': ['No', 'Yes']
    })
    
    processed = preprocess_data(mock_data, target_col='Churn')
    
    # Assertions
    assert 'customerID' not in processed.columns
    assert processed['MonthlyCharges'].dtype in [np.float64, np.int64]
    assert processed['TotalCharges'].dtype in [np.float64, np.int64]
    assert processed['Churn'].tolist() == [0, 1]
    # TotalCharges for second row should be filled with tenure * MonthlyCharges = 5 * 60.0 = 300.0
    assert processed['TotalCharges'].iloc[1] == 300.0

def test_build_features():
    # Mock raw data before preprocessing and dummification (3 rows to simulate high cardinality)
    mock_data = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'Partner': ['No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No'],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'MonthlyCharges': [50.0, 60.0, 70.0],
        'TotalCharges': [500.0, 300.0, 400.0],
        'tenure': [10, 5, 8],
        
        # Categorical multi-value columns (3 distinct values each to ensure nunique > 2)
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card (automatic)']
    })
    
    features = build_features(mock_data, target_col='Churn')
    
    # Assertions
    # Check that binary mapping occurred
    assert features['gender'].iloc[0] == 1  # Male -> 1
    assert features['gender'].iloc[1] == 0  # Female -> 0
    
    # Check that No_internet_service collapsed column exists
    assert 'No_internet_service' in features.columns
    
    # Assert original "No internet service" dummies were collapsed and removed
    for col in features.columns:
        assert 'No internet service' not in col

def test_inference_predict():
    # A single valid sample mimicking CustomerData pydantic schema
    sample = {
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0
    }
    
    prediction = predict(sample)
    assert prediction in ["Likely to churn", "Not likely to churn"]

if __name__ == "__main__":
    pytest.main([__file__])
