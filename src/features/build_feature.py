import pandas as pd

def _map_binary_cols(s:pd.Series) -> pd.Series:
    """ Apply Deterministic Mapping to Binary Columns 
    
    This function implements the core binary encoding logic that converts 
    categorical features with exactly two unique values into 0 and 1 integers.
    These mappings are deterministic ensuring consistency between training and serving. 
    """
    unique_vals = set(s.dropna().unique().astype(str))

    if unique_vals.issubset({'No', 'Yes'}):
        return s.map({'No': 0, 'Yes': 1}).astype('Int64')
    elif unique_vals.issubset({'Male', 'Female'}):
        return s.map({'Female': 0, 'Male': 1}).astype('Int64')
    
    vals = pd.Series(s.dropna().unique()).astype(str).tolist()
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        return s.astype(str).map({sorted_vals[0]: 0, sorted_vals[1]: 1}).astype('Int64')
    
    return s

def build_features(
    df: pd.DataFrame, 
    target_col: str = "Churn", 
    binary_cols: list = None, 
    multi_cols: list = None
) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """
    df = df.copy()
    print(f" Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f" Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # === STEP 2: Split Categorical by Cardinality ===
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    if binary_cols is None or multi_cols is None:
        binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
        multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f" Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f" Binary: {binary_cols}")
    if multi_cols:
        print(f" Multi-category: {multi_cols}")

    # === STEP 3: Apply Binary Encoding ===
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_cols(df[c])
        print(f"{c}: {original_dtype} → binary (0/1)")

    # === STEP 4: Convert Boolean Columns ===
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"   Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns")

        # === STEP 5.1: Redundancy / Collinearity Reduction (from EDA) ===
        # Combine redundant "No internet service" columns into a single indicator
        # and drop the individual redundant dummy columns.
        no_internet_cols = [c for c in df.columns if 'No internet service' in c]
        if no_internet_cols:
            df['No_internet_service'] = df[no_internet_cols].any(axis=1).astype(int)
            df = df.drop(columns=no_internet_cols)
            print(f"   Collapsed redundant 'No internet service' columns: {no_internet_cols}")

        # Handle PhoneService redundancy
        # 'MultipleLines_No phone service' is redundant with 'PhoneService' = 0
        if 'MultipleLines_No phone service' in df.columns:
            df = df.drop(columns=['MultipleLines_No phone service'])
            print("   Dropped redundant column 'MultipleLines_No phone service'")

    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df