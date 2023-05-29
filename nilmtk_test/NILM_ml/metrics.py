from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

regression_metrics = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "MAPE": mean_absolute_percentage_error,
    "R2": r2_score
}

classification_metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
}

def evaluate_regression(y_pred, y_true):
    """
    Evaluates predicted values against true values using regression metrics.

    Parameters:
        - y_pred (array-like): Predicted values.
        - y_true (array-like): True values.

    Returns:
        - mse (float): Mean Squared Error.
        - mae (float): Mean Absolute Error.
        - r2 (float): R-squared score.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    return mse, mae, r2