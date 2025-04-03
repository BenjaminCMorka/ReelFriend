"""
data loading + preprocessing for the recommender
"""
import os
import pandas as pd
import numpy as np

def load_movielens_data(data_path):
    """
    loads MovieLens data and preps it for training
    
    Args:
        data_path: path to the folder with the CSV files
    
    Returns:
        ratings_df and movies_df with all the extra features added
    """
    # get ratings
    ratings_path = os.path.join(data_path, 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)
    
    # get movies
    movies_path = os.path.join(data_path, 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    
    # try to pull the year out of the movie titles
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    
    # fallback for missing years
    median_year = movies_df['year'].median()
    movies_df['year'] = movies_df['year'].fillna(median_year)
    
    # split up genres into lists
    movies_df['genres'] = movies_df['genres'].str.split('|')
    genres_list = movies_df['genres'].explode().unique().tolist()
    if '(no genres listed)' in genres_list:
        genres_list.remove('(no genres listed)')
    genre_names = sorted(genres_list)
    
    # make one-hot columns for each genre
    for genre in genre_names:
        movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
    
    # count how many ratings each movie got
    movie_popularity = ratings_df.groupby('movieId').size().reset_index(name='movie_rating_count')
    movies_df = pd.merge(movies_df, movie_popularity, on='movieId', how='left')
    movies_df['movie_rating_count'] = movies_df['movie_rating_count'].fillna(0)
    
    # average rating for each movie
    movie_avg_rating = ratings_df.groupby('movieId')['rating'].mean().reset_index(name='movie_avg_rating')
    movies_df = pd.merge(movies_df, movie_avg_rating, on='movieId', how='left')
    movies_df['movie_avg_rating'] = movies_df['movie_avg_rating'].fillna(ratings_df['rating'].mean())
    
    # how active each user is
    user_activity = ratings_df.groupby('userId').size().reset_index(name='user_rating_count')
    ratings_df = pd.merge(ratings_df, user_activity, on='userId', how='left')
    
    # average rating each user gives
    user_avg_rating = ratings_df.groupby('userId')['rating'].mean().reset_index(name='user_avg_rating')
    ratings_df = pd.merge(ratings_df, user_avg_rating, on='userId', how='left')
    
    # attach movie info to the ratings
    ratings_df = pd.merge(
        ratings_df,
        movies_df[['movieId', 'year', 'movie_rating_count', 'movie_avg_rating']],
        on='movieId', how='left'
    )
    
    # break down timestamp into day/hour/month etc
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df['day_of_week'] = ratings_df['timestamp'].dt.dayofweek
    ratings_df['hour_of_day'] = ratings_df['timestamp'].dt.hour
    ratings_df['month'] = ratings_df['timestamp'].dt.month
    ratings_df['year_rated'] = ratings_df['timestamp'].dt.year
    
    # how recent the rating was (in days)
    max_timestamp = ratings_df['timestamp'].max()
    ratings_df['days_since_rated'] = (max_timestamp - ratings_df['timestamp']).dt.total_seconds() / (24 * 60 * 60)
    
    # how old the movie was when rated
    ratings_df['movie_age'] = ratings_df['year_rated'] - ratings_df['year']
    
    print(f"loaded {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users on {ratings_df['movieId'].nunique()} movies")
    print(f"found {len(genre_names)} unique genres")
    
    return ratings_df, movies_df
