import mlflow
import pandas as pd
import mlflow.xgboost
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_model(df: pd.DataFrame, target_col: str = "Churn") -> XGBClassifier:
    """
    Train an XGBoost model on the Telco Customer Churn dataset.
    
    This function implements the complete training pipeline including data splitting,
    model training, and evaluation. It also logs the model and metrics to MLflow for
    experiment tracking and reproducibility.
    
    """
    print(" Starting model training...")
    
    # === STEP 1: Prepare Data ===
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == 'object':
        y = y.str.strip().map({'Yes': 1, 'No': 0})  # Ensure target is binary (0/1)
    
    print(f" Data prepared with {X.shape[0]} samples and {X.shape[1]} features")
    
    # === STEP 2: Split Data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f" Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")

    # === STEP 3: Handle Class Imbalance with SMOTE ===
    print("   Applying SMOTE to address class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE: {y_train_smote.shape[0]} training samples with balanced classes")

    # === STEP 4: Train Model ===
    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    with mlflow.start_run(run_name="XGBoost_Churn_Model_Training"):
    
        model.fit(X_train_smote, y_train_smote )
        print(" Model training completed")
        
        # === STEP 5: Evaluate Model ===
        y_pred = model.predict(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f" Evaluation Metrics - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        # === STEP 6: Log to MLflow ===
        mlflow.xgboost.log_model(model, "xgb_churn_model")
        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1
        })
        
        train_ds = mlflow.data.from_pandas(df,source="training_data")
        mlflow.log_input(train_ds, "training_data")
        print(" Model and metrics logged to MLflow")
    
    return model