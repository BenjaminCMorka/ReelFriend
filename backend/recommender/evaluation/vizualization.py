"""
visualization for model evaluation and analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_evaluation_metrics(y_true, y_pred, errors, rmse, mae):
    """
    plot various evaluation metrics and visualizations for model performance.
    
    Args:
        y_true: Ground truth ratings
        y_pred: Predicted ratings
        errors: Prediction errors (y_true - y_pred)
        rmse: Root Mean Squared Error value
        mae: Mean Absolute Error value
    """
    plt.figure(figsize=(15, 12))
    
    # plot predicted vs actual values
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.1)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'True vs Predicted Ratings\nRMSE: {rmse:.4f}, MAE: {mae:.4f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # plot error distribution
    plt.subplot(2, 2, 2)
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # plot error vs. predicted rating
    plt.subplot(2, 2, 3)
    plt.scatter(y_pred, errors, alpha=0.1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Error')
    plt.title('Error vs Predicted Rating')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot error vs. actual rating
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, errors, alpha=0.1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Rating')
    plt.ylabel('Error')
    plt.title('Error vs Actual Rating')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.close()

def detailed_error_analysis(X_test, y_true, y_pred, errors, user_popularity, movie_popularity):
    """
    Perform detailed error analysis considering various factors.
    
    Args:
        X_test: Test data features (user_idx, movie_idx)
        y_true: Ground truth ratings
        y_pred: Predicted ratings
        errors: Prediction errors
        user_popularity: User popularity scores
        movie_popularity: Movie popularity scores
    """
    plt.figure(figsize=(15, 10))
    
    # error vs movie popularity
    plt.subplot(2, 2, 1)
    plt.scatter(movie_popularity, np.abs(errors), alpha=0.1)
    plt.xlabel('Movie Popularity')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Movie Popularity')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # error vs user activity
    plt.subplot(2, 2, 2)
    plt.scatter(user_popularity, np.abs(errors), alpha=0.1)
    plt.xlabel('User Activity')
    plt.ylabel('Absolute Error')
    plt.title('Error vs User Activity')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # error distribution by rating value
    plt.subplot(2, 2, 3)
    
    # group errors by rating
    unique_ratings = np.unique(y_true)
    error_by_rating = []
    
    for rating in unique_ratings:
        mask = y_true == rating
        if mask.sum() > 0:
            error_by_rating.append(np.abs(errors[mask]).mean())
        else:
            error_by_rating.append(0)
    
    plt.bar(unique_ratings, error_by_rating)
    plt.xlabel('Rating Value')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error by Rating Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # density plot of prediction errors
    plt.subplot(2, 2, 4)
    sns.kdeplot(errors, fill=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Error Density Plot')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('detailed_error_analysis.png')
    plt.close()

def plot_training_history(history):
    """
    Plot the training and validation loss curves 
    
    Args:
        history: keras training history object
    """
    plt.figure(figsize=(12, 10))
    
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Huber)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # plot MAE
    plt.subplot(2, 1, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()