import os
import json
import joblib
import pandas as pd
from src.data.preprocess import preprocess_data
from src.features.build_feature import build_features

# Find project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

MODEL = None
PREPROCESSING = None
FEATURE_COLUMNS = None

def load_artifacts():
    """
    Lazy load modeling and feature engineering artifacts.
    """
    global MODEL, PREPROCESSING, FEATURE_COLUMNS
    if MODEL is None:
        model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
        prep_path = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
        feat_path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run the pipeline first.")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessing file not found at {prep_path}. Please run the pipeline first.")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature columns file not found at {feat_path}. Please run the pipeline first.")
        
        MODEL = joblib.load(model_path)
        PREPROCESSING = joblib.load(prep_path)
        with open(feat_path, "r") as f:
            FEATURE_COLUMNS = json.load(f)

def predict(data_dict: dict) -> str:
    """
    Generate churn prediction for a single customer record.
    
    Parameters:
    data_dict (dict): A dictionary containing customer features.
    
    Returns:
    str: "Likely to churn" or "Not likely to churn".
    """
    # Ensure artifacts are loaded
    load_artifacts()
    
    # Convert incoming dict into a DataFrame
    df = pd.DataFrame([data_dict])
    
    target_col = PREPROCESSING.get("target", "Churn")
    binary_cols = PREPROCESSING.get("binary_cols")
    multi_cols = PREPROCESSING.get("multi_cols")
    
    # 1. Clean data (fix numeric types, fill missing TotalCharges)
    df = preprocess_data(df, target_col=target_col)
    
    # 2. Build engineered features
    df_enc = build_features(
        df, 
        target_col=target_col,
        binary_cols=binary_cols,
        multi_cols=multi_cols
    )
    
    # 3. Ensure boolean columns are converted to integers for XGBoost
    for col in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[col] = df_enc[col].astype(int)
        
    # 4. Reindex features to align with the training columns order exactly
    df_enc = df_enc.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    # 5. Run inference
    probability = MODEL.predict_proba(df_enc)[0, 1]
    
    # Classification threshold (0.35 optimized for churn detection recall/precision)
    threshold = 0.35
    
    if probability >= threshold:
        return "Likely to churn"
    else:
        return "Not likely to churn"
