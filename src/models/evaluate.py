from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):

    """
    Evaluates the performance of XGBoost model on the test set and prints the 
    classification report, confusion matrix, and ROC AUC score.

    Args:
        model: The trained XGBoost model to evaluate.
        X_test: The test set features.
        y_test: The true labels for the test set.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")