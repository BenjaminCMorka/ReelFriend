import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_movielens_data(data_path=""):
    """
    Load the MovieLens dataset and return the dataframes
    
    Args:
        data_path: Path to the MovieLens dataset
        
    Returns:
        ratings_df, movies_df: Pandas DataFrames containing ratings and movies data
    """
    print("Loading MovieLens data for exploration...")
    
    # oad ratings data
    ratings_path = os.path.join(data_path, 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)
    
    # load movies data
    movies_path = os.path.join(data_path, 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    
    print(f"Loaded {len(ratings_df)} ratings for {ratings_df['movieId'].nunique()} movies from {ratings_df['userId'].nunique()} users")
    
    return ratings_df, movies_df

def explore_ratings_distribution(ratings_df, save_path=None):
    """
    explore and visualize the distribution of ratings
    
    Args:
        ratings_df: DataFrame containing ratings data
        save_path: Path to save the visualizations, if None, plots are shown
    """
    plt.figure(figsize=(12, 8))
    
    # rating distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='rating', data=ratings_df, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # user activity distribution
    plt.subplot(2, 2, 2)
    user_ratings_count = ratings_df.groupby('userId').size()
    sns.histplot(user_ratings_count, bins=50, kde=True, color='purple')
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.xscale('log')
    
    # movie popularity distribution
    plt.subplot(2, 2, 3)
    movie_ratings_count = ratings_df.groupby('movieId').size()
    sns.histplot(movie_ratings_count, bins=50, kde=True, color='green')
    plt.title('Distribution of Ratings per Movie')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.xscale('log')
    
    # Rating distribution over time
    plt.subplot(2, 2, 4)
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_over_time = ratings_df.groupby(ratings_df['timestamp'].dt.to_period('M')).size()
    ratings_over_time.index = ratings_over_time.index.to_timestamp()
    plt.plot(ratings_over_time.index, ratings_over_time.values)
    plt.title('Ratings Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'ratings_distribution.png'))
    else:
        plt.show()

def explore_movie_features(movies_df, ratings_df, save_path=None):
    """
    explore and visualize movie features including genres and popularity
    
    Args:
        movies_df: DataFrame containing movies data
        ratings_df: DataFrame containing ratings data
        save_path: Path to save the visualizations, if None, plots are shown
    """
    # extract year from movie title
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    
    # calculate movie popularity (number of ratings)
    movie_popularity = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    movies_df = pd.merge(movies_df, movie_popularity, on='movieId', how='left')
    
    # calculate movie average rating
    movie_avg_rating = ratings_df.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
    movies_df = pd.merge(movies_df, movie_avg_rating, on='movieId', how='left')
    
    # explode genres for analysis
    movies_df['genres'] = movies_df['genres'].str.split('|')
    movies_exploded = movies_df.explode('genres')
    
    plt.figure(figsize=(16, 12))
    
    # movie count by year
    plt.subplot(2, 2, 1)
    year_counts = movies_df['year'].value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, alpha=0.7)
    plt.title('Number of Movies by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    
    # top genres
    plt.subplot(2, 2, 2)
    genre_counts = movies_exploded['genres'].value_counts().head(15)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
    plt.title('Top 15 Genres')
    plt.xlabel('Number of Movies')
    
    # ratings by year
    plt.subplot(2, 2, 3)
    year_ratings = movies_df.groupby('year')['avg_rating'].mean()
    plt.plot(year_ratings.index, year_ratings.values, marker='o', linestyle='-', alpha=0.7)
    plt.title('Average Rating by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # popularity vs. Rating
    plt.subplot(2, 2, 4)
    plt.scatter(movies_df['rating_count'], movies_df['avg_rating'], alpha=0.5)
    plt.title('Movie Popularity vs. Average Rating')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Average Rating')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'movie_features.png'))
    else:
        plt.show()
    
    # extra figure for genre analysis
    plt.figure(figsize=(14, 10))
    
    # average rating by genre
    plt.subplot(2, 1, 1)
    genre_ratings = movies_exploded.groupby('genres')['avg_rating'].mean().sort_values(ascending=False)
    sns.barplot(x=genre_ratings.index[:15], y=genre_ratings.values[:15], palette='viridis')
    plt.title('Average Rating by Genre (Top 15)')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    
    # popularity by genre
    plt.subplot(2, 1, 2)
    genre_popularity = movies_exploded.groupby('genres')['rating_count'].mean().sort_values(ascending=False)
    sns.barplot(x=genre_popularity.index[:15], y=genre_popularity.values[:15], palette='viridis')
    plt.title('Average Popularity by Genre (Top 15)')
    plt.xlabel('Genre')
    plt.ylabel('Average Number of Ratings')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'genre_analysis.png'))
    else:
        plt.show()

def analyze_user_behavior(ratings_df, save_path=None):
    """
    analyze and visualize user behavior patterns
    
    Args:
        ratings_df: DataFrame containing ratings data
        save_path: Path to save the visualizations, if None, plots are shown
    """
    # convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # extract temporal features
    ratings_df['day_of_week'] = ratings_df['timestamp'].dt.dayofweek
    ratings_df['hour_of_day'] = ratings_df['timestamp'].dt.hour
    ratings_df['month'] = ratings_df['timestamp'].dt.month
    ratings_df['year'] = ratings_df['timestamp'].dt.year
    
    plt.figure(figsize=(16, 12))
    
    # ratings by day of week
    plt.subplot(2, 2, 1)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = ratings_df['day_of_week'].value_counts().sort_index()
    plt.bar(day_names, day_counts.values, alpha=0.7)
    plt.title('Ratings by Day of Week')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    
    # ratings by hour of day
    plt.subplot(2, 2, 2)
    hour_counts = ratings_df['hour_of_day'].value_counts().sort_index()
    plt.bar(hour_counts.index, hour_counts.values, alpha=0.7)
    plt.title('Ratings by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Ratings')
    
    # average rating by day of week
    plt.subplot(2, 2, 3)
    day_avg_ratings = ratings_df.groupby('day_of_week')['rating'].mean()
    plt.bar(day_names, day_avg_ratings.values, alpha=0.7)
    plt.title('Average Rating by Day of Week')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    
    # average rating by hour of day
    plt.subplot(2, 2, 4)
    hour_avg_ratings = ratings_df.groupby('hour_of_day')['rating'].mean()
    plt.bar(hour_avg_ratings.index, hour_avg_ratings.values, alpha=0.7)
    plt.title('Average Rating by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Average Rating')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'user_behavior.png'))
    else:
        plt.show()
    
    # create user engagement statistics
    user_stats = ratings_df.groupby('userId').agg({
        'rating': ['count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    })
    
    user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'first_rating', 'last_rating']
    user_stats['days_active'] = (user_stats['last_rating'] - user_stats['first_rating']).dt.days
    user_stats['ratings_per_day'] = user_stats['rating_count'] / user_stats['days_active'].replace(0, 1)
    
    plt.figure(figsize=(14, 10))
    
    # user engagement histogram
    plt.subplot(2, 2, 1)
    sns.histplot(user_stats['days_active'], bins=50, kde=True)
    plt.title('User Engagement Period Distribution')
    plt.xlabel('Days Active')
    plt.ylabel('Number of Users')
    plt.xscale('log')
    
    # ratings per day histogram
    plt.subplot(2, 2, 2)
    sns.histplot(user_stats['ratings_per_day'].clip(0, 10), bins=50, kde=True)
    plt.title('Ratings Per Day Distribution')
    plt.xlabel('Ratings Per Day')
    plt.ylabel('Number of Users')
    
    # user rating variance
    plt.subplot(2, 2, 3)
    sns.histplot(user_stats['rating_std'].dropna(), bins=50, kde=True)
    plt.title('User Rating Standard Deviation')
    plt.xlabel('Standard Deviation of Ratings')
    plt.ylabel('Number of Users')
    
    # average rating distribution
    plt.subplot(2, 2, 4)
    sns.histplot(user_stats['avg_rating'], bins=50, kde=True)
    plt.title('User Average Rating Distribution')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'user_engagement.png'))
    else:
        plt.show()
    
    return user_stats

def run_complete_exploration(data_path="", save_path="visualization_results"):
    """
    Run a complete exploratory data analysis and save all visualizations
    
    Args:
        data_path: Path to the MovieLens dataset
        save_path: Path to save all visualizations
    """
    # create directory for saving visualizations if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # load data
    ratings_df, movies_df = load_movielens_data(data_path)
    
    # run all explorations
    explore_ratings_distribution(ratings_df, save_path)
    explore_movie_features(movies_df, ratings_df, save_path)
    user_stats = analyze_user_behavior(ratings_df, save_path)
    
    # save summary statistics
    ratings_summary = ratings_df.describe()
    ratings_summary.to_csv(os.path.join(save_path, 'ratings_summary.csv'))
    
    user_stats.describe().to_csv(os.path.join(save_path, 'user_stats_summary.csv'))
    print(f"Exploration complete. Results saved to {save_path}")
    return ratings_df, movies_df

if __name__ == "__main__":

    data_path = "data/"  
    
    run_complete_exploration(data_path)