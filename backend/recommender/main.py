"""
Main module for running the recommender system.
"""
import os
import json
import time
import argparse
import sys
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.recommendation_model import Recommender
from utils.id_mapping import map_tmdb_to_movielens, map_movielens_to_tmdb
from utils.utils import save_experiment_results
from explanation.shap_explanations import print_recommendation_explanation

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# fallback recommendations based on input movies' genres
def get_genre_specific_recommendations(recommender, favorite_movies):
    genres = set()
    for movie_id in favorite_movies:
        movie_row = recommender.movies_df[recommender.movies_df["movieId"] == int(movie_id)]
        if not movie_row.empty:
            genres.update([g for g in recommender.genre_names if movie_row[g].values[0] == 1])
    
    # filter movies by these genres and sort by rating count
    genre_movies = recommender.movies_df[
        recommender.movies_df[list(genres)].sum(axis=1) > 0
    ].sort_values('movie_rating_count', ascending=False).head(24)
    
    return [
        (row['title'], 0, "|".join([g for g in recommender.genre_names if row[g] == 1])) 
        for _, row in genre_movies.iterrows()
    ]
def main(data_path="data", mode="load", model_path="saved_model", 
         tune_hyperparameters=False, favorite_movies=None, user_id=None, new_ratings=None):
    """
    Main function to run the recommender system.
    
    Args:
        data_path: Path to the dataset directory
        mode: 'train' to train a new model, 'load' to load existing model, 'update_embedding' to update user embedding
        model_path: Path to the model directory when loading an existing model
        tune_hyperparameters: Whether to run hyperparameter tuning
        favorite_movies: List of favorite movie IDs for new user recommendations
        user_id: ID of the user
        new_ratings: JSON string containing new ratings for embedding updates
    """
    print("MovieLens Recommender System")
    print("-" * 40)
    
    start_time = time.time()
    
    # find data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_data_dirs = [
        os.path.join(script_dir, data_path),
        os.path.join(script_dir, '..', data_path),
        os.path.join(script_dir, '..', '..', data_path)
    ]
    
    # find the first directory that contains ratings.csv
    data_dir = None
    for possible_dir in possible_data_dirs:
        if os.path.exists(os.path.join(possible_dir, 'ratings.csv')):
            data_dir = possible_dir
            data_path = possible_dir  
            print(f"Found data directory at: {data_dir}")
            break
    
    if not data_dir:
        print(f"Error: couldn't find ratings.csv.")
        # use fallback recs if data can't be found
        fallback_results = {
            "recommendations": ["299534", "299536", "24428", "299537", "10138", "76341", "100402", "284053", "118340", "245891"],
            "explanations": ["Recommended based on your favorite movies"] * 10
        }
        
        # write fallback results to a file
        with open('recommendation_results.json', 'w') as f:
            json.dump(fallback_results, f)
        
        # print the fallback results to stdout
        print("\nRESULTS_JSON_START")
        print(json.dumps(fallback_results))
        print("RESULTS_JSON_END")
        print("\nUsing fallback recommendations due to missing data files.")
        return None
    
    # map TMDB ids to MovieLens ids
    if favorite_movies:
        favorite_movies_ml = map_tmdb_to_movielens(favorite_movies, data_path)
        print(f"Mapped TMDB IDs to MovieLens IDs: {favorite_movies} -> {favorite_movies_ml}")
        favorite_movies = favorite_movies_ml
    
    # Load optimal model parameters if json for them is available
    optimal_params = {
        "embedding_dim": 128,
        "batch_size": 128,
        "learning_rate": 0.0005,
        "dropout_rate": 0.3,
        "l2_reg": 0.00005,
        "use_bias": True,
        "use_features": True,
        "use_time_features": True
    } # defining optimal params incase optimal json is not available
    
    # try to load params from JSON file
    params_file = os.path.join("optimal_model_results", "model_params.json")
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                loaded_params = json.load(f)
                optimal_params.update(loaded_params)
                print(f"Loaded optimal model parameters from {params_file}")
        except Exception as e:
            print(f"Error loading optimal parameters: {e}")
    
    if mode == "tune" or tune_hyperparameters:
        print("Running hyperparameter tuning...")
        try:
            from hyperparameter_tuning import efficient_hyperparameter_tuning
            best_config = efficient_hyperparameter_tuning(data_path, Recommender, n_trials=20)
            print("Hyperparameter tuning completed.")
            print(f"Best configuration: {best_config}")
            
            # save the best configuration
            os.makedirs("optimal_model_results", exist_ok=True)
            with open(os.path.join("optimal_model_results", "model_params.json"), "w") as f:
                json.dump(best_config, f, indent=2)
        except ImportError:
            print("hyperparameter tuning module not found. skipping tuning.")
            
    elif mode == "train":
        print("Training a new recommendation model...")
        print(f"Using optimal parameters: {optimal_params}")
        recommender = Recommender(
            data_path=data_path,
            embedding_dim=optimal_params["embedding_dim"],
            batch_size=optimal_params["batch_size"],
            learning_rate=optimal_params["learning_rate"],
            dropout_rate=optimal_params["dropout_rate"],
            l2_reg=optimal_params["l2_reg"],
            use_bias=optimal_params["use_bias"],
            use_features=optimal_params["use_features"],
            use_time_features=optimal_params["use_time_features"]
        )
        
        # load and prepare data
        recommender.load_data()
        recommender.prepare_train_val_test_split()
        
        # build and train model
        recommender.build_model()
        recommender.train(epochs=100, patience=10)
        
        # evaluate the model
        rmse, mae, precision_5, recall_5 = recommender.evaluate()
        metrics_results = recommender.evaluate_ranking_metrics(k_values=[1, 3, 5, 10])

        
        # plot training history
        recommender.plot_training_history()
        
        # save  model
        model_path = "saved_model"
        recommender.save_model(model_path)
        
        # save experiment results
        model_info = {
            "embedding_dim": recommender.embedding_dim,
            "batch_size": recommender.batch_size,
            "learning_rate": recommender.learning_rate,
            "dropout_rate": recommender.dropout_rate,
            "l2_reg": recommender.l2_reg,
            "use_bias": recommender.use_bias,
            "use_features": recommender.use_features,
            "use_time_features": recommender.use_time_features
        }
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "training_time": time.time() - start_time
        }
        
        
        save_experiment_results(model_info, metrics)
    
    elif mode == "update_embedding":
        print(f"Loading model and updating user embedding for user {user_id}...")
        # create a recommender with the optimal parameters
        recommender = Recommender(
            data_path=data_path,
            embedding_dim=optimal_params["embedding_dim"],
            batch_size=optimal_params["batch_size"],
            learning_rate=optimal_params["learning_rate"],
            dropout_rate=optimal_params["dropout_rate"],
            l2_reg=optimal_params["l2_reg"],
            use_bias=optimal_params["use_bias"],
            use_features=optimal_params["use_features"],
            use_time_features=optimal_params["use_time_features"]
        )
        
        # Load data first for reference
        recommender.load_data()
        
        # load  pre-trained model
        success = recommender.load_model(model_path)
        if not success:
            print("Failed to load model. Cannot update embeddings.")
            return None
            
        # parse new ratings
        try:
            if new_ratings:
                parsed_ratings = json.loads(new_ratings)
                print(f"Updating user {user_id} embedding with {len(parsed_ratings)} new ratings")
                
                # convert TMDB ids to MovieLens ids for each new rating
                ml_ratings = []
                for rating in parsed_ratings:
                    tmdb_id = rating['movieId']
                    ml_id = map_tmdb_to_movielens([tmdb_id], data_path)[0]
                    if ml_id:
                        ml_ratings.append({
                            'movie_id': ml_id,
                            'rating': rating['rating']
                        })
                
                # update user embedding with new ratings
                for rating in ml_ratings:
                    recommender.update_user_embedding(user_id, rating)
                    print(f"Updated embedding for movie {rating['movie_id']} with rating {rating['rating']}")
        except Exception as e:
            print(f"Error updating user embedding: {e}")
            import traceback
            traceback.print_exc()
    
    else:  # default is load mode
        print(f"Loading existing model from {model_path}...")
        # create recommender with the optimal parameters
        recommender = Recommender(
            data_path=data_path,
            embedding_dim=optimal_params["embedding_dim"],
            batch_size=optimal_params["batch_size"],
            learning_rate=optimal_params["learning_rate"],
            dropout_rate=optimal_params["dropout_rate"],
            l2_reg=optimal_params["l2_reg"],
            use_bias=optimal_params["use_bias"],
            use_features=optimal_params["use_features"],
            use_time_features=optimal_params["use_time_features"]
        )
        
        # Just load the data (needed for recommendations)
        recommender.load_data()
        
        # Load the pre-trained model
        success = recommender.load_model(model_path)
        if not success:
            print("Failed to load model. Cannot provide recommendations.")
            return None
    
    # get recommendations with explanations
    print("\nRecommendations with natural language explanations:")

    new_user_id = user_id
    new_user_ratings = []

    for movie_id in favorite_movies:
        new_user_ratings.append({
            'userId': new_user_id,
            'movieId': movie_id,
            'rating': 4.0,
            'timestamp': int(time.time())
        })
    

    new_ratings_df = pd.DataFrame(new_user_ratings)
    recommender.ratings_df = recommender.ratings_df[recommender.ratings_df['userId'] != new_user_id]
    recommender.ratings_df = pd.concat([recommender.ratings_df, new_ratings_df], ignore_index=True)
    recommender.load_data()
    favorite_movie_ids = favorite_movies  # use  mapped MovieLens IDs
    
    # print the favorite movies for reference
    print(f"\nuser ({user_id}) favorite movies:")
    for i, movie_id in enumerate(favorite_movie_ids):
        # convert movie_id to int for lookup
        try:
            movie_id_int = int(movie_id)
        except (ValueError, TypeError):
            print(f"Warning: Invalid movie ID format: {movie_id}")
            continue
            
        movie_row = recommender.movies_df[recommender.movies_df["movieId"] == movie_id_int]
        if not movie_row.empty:
            title = movie_row["title"].values[0]
            genres = "|".join([g for g in recommender.genre_names if movie_row[g].values[0] == 1])
            print(f"{i+1}. {title} - {genres}")
        else:
            print(f"{i+1}. Movie ID {movie_id} not found in dataset")
    
    # generate recommendations
    recommendations = recommender.recommend_for_user(
        user_id=new_user_id,
        favorite_movie_ids=favorite_movie_ids,
        top_n=24,
        diversity_factor=0.2  # Reduced diversity for more relevant recommendations
    )
    
    # get and show explanations
    explanations = print_recommendation_explanation(recommender, new_user_id, recommendations)

    # convert MovieLens ids back to TMDB ids for the frontend
    movie_ids = []
    for rec in recommendations:
        try:
            # try to find the MovieLens id for title
            movie_row = recommender.movies_df[recommender.movies_df["title"] == rec[0]]
            if not movie_row.empty:
                movie_ids.append(movie_row["movieId"].values[0])
            else:
                print(f"Could not find MovieLens ID for recommendation: {rec[0]}")
        except Exception as e:
            print(f"Error extracting MovieLens ID for recommendation: {e}")
    
    tmdb_recommendations = map_movielens_to_tmdb(movie_ids, data_path)
    
    # create results dictionary
    results = {
        "recommendations": tmdb_recommendations,
        "explanations": explanations
    }
    
    # write results to file that can be read by controller
    with open('recommendation_results.json', 'w') as f:
        json.dump(results, f)
    
    # print the results to stdout
    print("\nRESULTS_JSON_START")
    print(json.dumps(results))
    print("RESULTS_JSON_END")

    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
    print("\nDone!")
    return recommender


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Recommender System')
    parser.add_argument('--data-path', type=str, default='data',
                        help='path to the data directory')
    parser.add_argument('--mode', type=str, default='load', choices=['train', 'load', 'tune', 'update_embedding'],
                        help='mode to run the recommender')
    parser.add_argument('--model-path', type=str, default='saved_model',
                        help='path to the saved model directory')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='whether to run hyperparameter tuning')
    parser.add_argument('--favorite-movies', type=str, default=None,
                        help='list of favorite movie IDs for new user recommendations')
    parser.add_argument('--user-id', type=str, default=None,
                        help='User ID for personalized training')
    parser.add_argument('--new-ratings', type=str, default=None,
                        help='JSON string with new ratings for embedding updates')
    
    args = parser.parse_args()
    
    # parse favorite movies 
    favorite_movies = None
    if args.favorite_movies:
        favorite_movies = [id.strip() for id in args.favorite_movies.split(',') if id.strip()]
        
    main(
        data_path=args.data_path,
        mode=args.mode,
        model_path=args.model_path,
        tune_hyperparameters=args.tune_hyperparameters,
        favorite_movies=favorite_movies,
        user_id=args.user_id,
        new_ratings=args.new_ratings
    )