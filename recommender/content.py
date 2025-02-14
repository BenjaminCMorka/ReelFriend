import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

class LightweightMovieRecommender:
    def __init__(self, movies_path='data/movies_processed.csv'):
        logging.info("Loading movie data...")
        self.movies = pd.read_csv(movies_path)
        
        # Get genre columns
        self.genre_columns = [col for col in self.movies.columns 
                            if col not in ['movieId', 'title', 'genres', 'year']]
        
        # Create sparse matrix once
        self.movie_features = csr_matrix(self.movies[self.genre_columns].values)
        
        # Create title to index mapping for faster lookups
        self.title_to_idx = {title: idx for idx, title in enumerate(self.movies['title'])}
    
    def get_movie_recommendations(self, movie_title, n_recommendations=10):
        """Calculate similarities on-the-fly only for the requested movie"""
        # Get movie index
        idx = self.title_to_idx.get(movie_title)
        if idx is None:
            return "Movie not found in the dataset."
        
        # Get the feature vector for this movie
        movie_vector = self.movie_features[idx:idx+1]
        
        # Calculate similarities only between this movie and all others
        similarities = cosine_similarity(movie_vector, self.movie_features).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        # Get the movie titles and their similarity scores
        recommendations = pd.DataFrame({
            'title': self.movies['title'].iloc[similar_indices],
            'similarity': similarities[similar_indices]
        })
        
        return recommendations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recommender = LightweightMovieRecommender()
    
    # Get recommendations
    movie_title = 'Willy Wonka & the Chocolate Factory (1971)'
    recommendations = recommender.get_movie_recommendations(movie_title)
    print(f"\nRecommendations for {movie_title}:")
    print(recommendations)