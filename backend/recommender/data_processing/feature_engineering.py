"""
prepares data for training the recommender — does scaling, feature prep, and splitting
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_train_val_test_split(ratings_df, movies_df, n_genres, genre_names, 
                              rating_min, rating_max, val_size=0.1, test_size=0.1, 
                              stratify_recent=True):
    """
    splits the data into training, validation, and test sets. 
    does optional stratification so recent/older ratings get spread evenly.
    
    Args:
        ratings_df: the full ratings dataframe
        movies_df: movie metadata
        n_genres: total number of genres
        genre_names: list of all genre names
        rating_min: minimum rating in the dataset
        rating_max: maximum rating in the dataset
        val_size: fraction for validation set
        test_size: fraction for test set
        stratify_recent: whether to balance recent/older ratings across splits
        
    Returns:
        the 3 splits (train, val, test) + scalers for user/movie/time features
    """
    
    print("Preparing train/validation/test splits...")

    # user and movie indices
    X = ratings_df[['user_idx', 'movie_idx']].values

    # get raw popularity and rating stats
    user_popularity = ratings_df[['user_rating_count']].values
    movie_popularity = ratings_df[['movie_rating_count']].values
    user_avg = ratings_df[['user_avg_rating']].values
    movie_avg = ratings_df[['movie_avg_rating']].values

    # scale popularity features to [0, 1]
    user_popularity_scaler = MinMaxScaler()
    movie_popularity_scaler = MinMaxScaler()
    user_popularity_scaled = user_popularity_scaler.fit_transform(user_popularity)
    movie_popularity_scaled = movie_popularity_scaler.fit_transform(movie_popularity)

    # normalize average rating to 0-1 scale
    user_avg_scaled = (user_avg - rating_min) / (rating_max - rating_min)
    movie_avg_scaled = (movie_avg - rating_min) / (rating_max - rating_min)

    # grab and scale time-based features
    time_features = ratings_df[['day_of_week', 'hour_of_day', 'month', 'days_since_rated', 'movie_age']].values
    time_scaler = StandardScaler()
    time_features_scaled = time_scaler.fit_transform(time_features)

    # target variable is the actual rating
    y = ratings_df['rating'].values

    # grab genre info for each movie in the ratings
    genre_features = np.zeros((len(ratings_df), n_genres))
    for i, row in enumerate(ratings_df.itertuples()):
        movie_id = row.movieId
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            genre_features[i] = movie_row[genre_names].values[0]

    # if enabled, stratify by recency 
    if stratify_recent:
        recent_threshold = np.percentile(ratings_df['days_since_rated'], 25)
        is_recent = (ratings_df['days_since_rated'] <= recent_threshold).astype(int)

        # do the first split
        X_train, X_temp, y_train, y_temp, genre_train, genre_temp, time_train, time_temp, \
        user_pop_train, user_pop_temp, movie_pop_train, movie_pop_temp, \
        user_avg_train, user_avg_temp, movie_avg_train, movie_avg_temp = train_test_split(
            X, y, genre_features, time_features_scaled, 
            user_popularity_scaled, movie_popularity_scaled,
            user_avg_scaled, movie_avg_scaled,
            test_size=val_size + test_size, stratify=is_recent, random_state=42)

        # split temp into val and test
        X_val, X_test, y_val, y_test, genre_val, genre_test, time_val, time_test, \
        user_pop_val, user_pop_test, movie_pop_val, movie_pop_test, \
        user_avg_val, user_avg_test, movie_avg_val, movie_avg_test = train_test_split(
            X_temp, y_temp, genre_temp, time_temp, 
            user_pop_temp, movie_pop_temp,
            user_avg_temp, movie_avg_temp,
            test_size=0.5, random_state=42)

    else:
        # normal split without stratifying by recency
        X_train, X_temp, y_train, y_temp, genre_train, genre_temp, time_train, time_temp, \
        user_pop_train, user_pop_temp, movie_pop_train, movie_pop_temp, \
        user_avg_train, user_avg_temp, movie_avg_train, movie_avg_temp = train_test_split(
            X, y, genre_features, time_features_scaled, 
            user_popularity_scaled, movie_popularity_scaled,
            user_avg_scaled, movie_avg_scaled,
            test_size=val_size + test_size, random_state=42)

        X_val, X_test, y_val, y_test, genre_val, genre_test, time_val, time_test, \
        user_pop_val, user_pop_test, movie_pop_val, movie_pop_test, \
        user_avg_val, user_avg_test, movie_avg_val, movie_avg_test = train_test_split(
            X_temp, y_temp, genre_temp, time_temp, 
            user_pop_temp, movie_pop_temp,
            user_avg_temp, movie_avg_temp,
            test_size=0.5, random_state=42)

    # bundle everything
    train_data = (X_train, y_train, genre_train, time_train, 
                  user_pop_train, movie_pop_train, user_avg_train, movie_avg_train)
    val_data = (X_val, y_val, genre_val, time_val, 
                user_pop_val, movie_pop_val, user_avg_val, movie_avg_val)
    test_data = (X_test, y_test, genre_test, time_test, 
                 user_pop_test, movie_pop_test, user_avg_test, movie_avg_test)

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return train_data, val_data, test_data, user_popularity_scaler, movie_popularity_scaler, time_scaler
