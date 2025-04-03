"""
Evaluation metrics for the recommender system.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    calculate standard evaluation metrics 
    
    Args:
        y_true: ground truth ratings
        y_pred: predicted ratings
        
    Returns:
        rmse and mae tuple
    """
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    return rmse, mae