import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import os

# global variables to store information across trials
_base_recommender = None
_base_data_loaded = False
_best_rmse = float('inf')
_best_mae = float('inf')
_best_config = None

def objective(trial, data_path, Recommender):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        data_path: Path to the dataset
        Recommender: The Recommender class to use
        
    Returns:
        RMSE value for this trial
    """
    global _base_recommender, _base_data_loaded, _best_rmse, _best_mae, _best_config
    
    # define the hyperparameters to search
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    
    # create a recommender with these hyperparameters
    recommender = Recommender(
        data_path=data_path,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        use_bias=True,  # Fixed parameter
        batch_size=batch_size,
        use_features=True,  # Fixed parameter
        use_time_features=True  # Fixed parameter
    )
    
    # load and prepare data
    if not _base_data_loaded:
        recommender.load_data()
        global _base_recommender
        _base_recommender = recommender
        _base_data_loaded = True
    else:
        # reuse data loading from previous trials
        recommender.ratings_df = _base_recommender.ratings_df.copy()
        recommender.movies_df = _base_recommender.movies_df.copy()
        recommender.n_users = _base_recommender.n_users
        recommender.n_movies = _base_recommender.n_movies
        recommender.n_genres = _base_recommender.n_genres
        recommender.genre_names = _base_recommender.genre_names
        recommender.rating_min = _base_recommender.rating_min
        recommender.rating_max = _base_recommender.rating_max
        recommender.user_id_map = _base_recommender.user_id_map.copy()
        recommender.movie_id_map = _base_recommender.movie_id_map.copy()
        recommender.inv_user_id_map = _base_recommender.inv_user_id_map.copy()
        recommender.inv_movie_id_map = _base_recommender.inv_movie_id_map.copy()
    
    # split, build, train, evaluate
    recommender.prepare_train_val_test_split()
    recommender.build_model()
    recommender.train(epochs=20, patience=5)  # Reduce epochs for faster tuning
    rmse, mae = recommender.evaluate()
    
    # report results back to Optuna
    trial.report(rmse, step=1)
    
    # save best model so far
    if rmse < _best_rmse:
        _best_rmse = rmse
        _best_mae = mae
        _best_config = {
            'embedding_dim': embedding_dim,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'l2_reg': l2_reg,
            'batch_size': batch_size,
            'rmse': rmse,
            'mae': mae
        }
        # Save the best model
        recommender.save_model(f"best_tuned_model_rmse_{rmse:.4f}")
    
    return rmse


def efficient_hyperparameter_tuning(data_path, Recommender, n_trials=20):
    """
    Perform hyperparameter tuning using Optuna.
    
    Args:
        data_path: Path to the dataset
        Recommender: The Recommender class to use
        n_trials: Number of trials to run
        
    Returns:
        Best hyperparameter configuration found
    """
    global _base_data_loaded, _best_rmse, _best_mae, _best_config
    
    # initialize best values
    _best_rmse = float('inf')
    _best_mae = float('inf')
    _best_config = None
    _base_data_loaded = False
    
    # create an Optuna study
    sampler = TPESampler(seed=42)  # TPE algorithm with reproducibility
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # run optimization
    print("Starting Optuna hyperparameter optimization...")
    study.optimize(lambda trial: objective(trial, data_path, Recommender), n_trials=n_trials)
    
    # report best results
    print("\n\nBest hyperparameters found:")
    print(study.best_params)
    
    print(f"Best RMSE: {_best_rmse:.4f}, Best MAE: {_best_mae:.4f}")
    
    # visualize results
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image("optimization_history.png")
        
        # plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image("parameter_importances.png")
    except:
        print("Couldn't generate visualizations - ensure plotly is installed")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_hyperparameter_results.csv", index=False)
    
    return _best_config