import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic preprocessing steps for the Telco Churn dataset:
    - Drop customerID column
    - Convert TotalCharges to numeric and handle missing values
    - Encode target variable (Churn) to binary

    """
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace from column names

    # Drop customerID as it is not useful for modeling
    for col in ['customerID', 'CustomerID', "customer_id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            break
    
    # Encode target variable (Churn) to binary
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].str.strip().map({'Yes': 1, 'No': 0})

    # Convert MonthlyCharges and tenure to numeric to avoid string/type mismatches
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Convert TotalCharges to numeric, coerce errors to NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with tenure * MonthlyCharges if numeric columns exist
    if 'TotalCharges' in df.columns and 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    
    # Encode Senior Citizen to binary
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)  # Fill any remaining numeric NaNs with 0

    
    return df