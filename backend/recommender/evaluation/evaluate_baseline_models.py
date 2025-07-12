import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import random

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

def load_movielens_data(data_path='data'):
    ratings_path = os.path.join('data', 'ratings.csv')
    
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}")
    
    # read ratings
    ratings_df = pd.read_csv(ratings_path)
    
    print(f"Loaded {len(ratings_df)} ratings")
    print(f"Number of unique users: {ratings_df['userId'].nunique()}")
    print(f"Number of unique movies: {ratings_df['movieId'].nunique()}")
    
    return ratings_df
def evaluate_recommender_model(trainset, testset, model, model_name):
    # fit model
    recommender = model
    recommender.fit(trainset)

    # predict on test set
    predictions = recommender.test(testset)

    # calculate metrics
    rmse_score = rmse(predictions)
    mae_score = mae(predictions)

    # precision/Recall at K  using 3.5 good rating
    threshold = 3.5 
    user_predictions = {}
    for pred in predictions:
        user_predictions.setdefault(pred.uid, []).append(pred)

    precisions, recalls = [], []
    for preds in user_predictions.values():
        preds.sort(key=lambda x: x.est, reverse=True)
        top_k = preds[:5]
        good_actuals = [p for p in preds if p.r_ui >= threshold]
        good_preds = [p for p in top_k if p.r_ui >= threshold]

        precisions.append(len(good_preds) / 5)
        recalls.append(len(good_preds) / len(good_actuals) if good_actuals else 0)

    return {
        'Model': model_name,
        'RMSE': rmse_score,
        'MAE': mae_score,
        'Precision@5': np.mean(precisions),
        'Recall@5': np.mean(recalls)
    }

def main():
    ratings_df = load_movielens_data()
    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)


    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    models = [
        (SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02), 'SVD'),
        (KNNBasic(sim_options={'name': 'cosine', 'user_based': True}, k=50, min_k=3), 'User-Based CF'),
        (KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, k=50, min_k=3), 'Item-Based CF')
    ]

    results = [evaluate_recommender_model(trainset, testset, model, name) for model, name in models]
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation/baseline_models_evaluation.csv', index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()