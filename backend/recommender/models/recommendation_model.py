import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from datetime import datetime
import pickle
from models.model_builder import build_recommender_model
from models.model_builder import build_recommender_model
from models.model_builder import zeros_like_function, zeros_like_output_shape
from models.model_builder import scale_by_factor, scale_by_factor_output_shape
from evaluation.vizualization import plot_evaluation_metrics, detailed_error_analysis


os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

CUSTOM_OBJECTS = {
    'zeros_like_function': zeros_like_function,
    'zeros_like_output_shape': zeros_like_output_shape,
    'scale_by_factor': scale_by_factor,
    'scale_by_factor_output_shape': scale_by_factor_output_shape
}

class Recommender:
    def __init__(self, data_path="data", embedding_dim=128, batch_size=128, 
                 learning_rate=0.0005, dropout_rate=0.3, l2_reg=0.00005, use_bias=True,
                 use_features=True, use_time_features=True):
        """
        Initialize the recommender system.
        
        Args:
            data_path: Path to the MovieLens dataset
            embedding_dim: Dimension of embeddings for users and items
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            use_bias: Whether to use bias terms in the model
            use_features: Whether to use movie features in the model
            use_time_features: Whether to use temporal features
        """
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.use_features = use_features
        self.use_time_features = use_time_features
        self.model = None
        self.history = None
        
     
        self.ratings_df = None
        self.movies_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
   
        self.n_users = None
        self.n_movies = None
        self.n_genres = None
        self.genre_names = None
        

        self.rating_min = None
        self.rating_max = None
        
        # training stats
        self.best_rmse = float('inf')
        self.best_mae = float('inf')
        

        self.time_scaler = None
        self.user_popularity_scaler = None
        self.movie_popularity_scaler = None

    def recommend_for_user(self, user_id, favorite_movie_ids=None, top_n=24, diversity_factor=0.4):
        """
        Generate recommendations for a user, handling both existing and new users.
        For new users, estimates user latent factors based on their favorite movies.
        
        Args:
            user_id: ID of the user (can be a new user not seen during training)
            favorite_movie_ids: List of movie IDs the user likes (for cold start)
            top_n: Number of recommendations to generate
            diversity_factor: Balance between accuracy and diversity (0-1)
            
        Returns:
            List of (title, predicted_rating, genres) tuples
        """
        # Check if this is an existing user in our system
        is_new_user = user_id not in self.user_id_map
        
        if is_new_user and (favorite_movie_ids is None or len(favorite_movie_ids) == 0):
            print(f"New user {user_id} without favorite movies. Using popular recommendations.")
            # Fall back to popularity-based recommendations
            popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
            return [(row['title'], 0.0, "|".join([g for g in self.genre_names if row[g] == 1])) 
                    for _, row in popular_movies.iterrows()]
        
        if is_new_user:
            print(f"New user {user_id} with {len(favorite_movie_ids)} favorite movies. Inferring preferences.")
            # get embeddings for their favorite movies and infer user latent factors
            return self._recommend_for_new_user(user_id, favorite_movie_ids, top_n, diversity_factor)
        else:
            # if existing user use the regular recommendation logic
            return self.get_recommendations(user_id, top_n, diversity_factor)
        
    def _recommend_for_new_user(self, user_id, favorite_movie_ids, top_n=24, diversity_factor=0.4):
        """
        Generate recommendations for a new user based on their favorite movies.

        Args:
            user_id: ID for the new user
            favorite_movie_ids: List of movie IDs the user likes
            top_n: Number of recommendations to generate
            diversity_factor: balance between accuracy and diversity (0-1)
            
        Returns:
            List of (title, predicted_rating, genres) tuples
        """
        #  map favorite movie ids to internal indices
        valid_movie_ids = []
        valid_indices = []

        for movie_id in favorite_movie_ids:
            if movie_id in self.movie_id_map:
                valid_movie_ids.append(movie_id)
                valid_indices.append(self.movie_id_map[movie_id])

        if not valid_indices:
            print("None of the provided favorite movies were found in the model.")
            # fallback to popularity-based recommendations
            popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
            return [(row['title'], 0.0, "|".join([g for g in self.genre_names if row[g] == 1])) 
                    for _, row in popular_movies.iterrows()]

        # Get movie embeddings layer from the model
        movie_embedding_layer = self.model.get_layer('movie_embedding')
        movie_embeddings = movie_embedding_layer.get_weights()[0]

        # get embeddings for the user's favorite movies
        favorite_embeddings = movie_embeddings[valid_indices]

        # infer user's latent factors by avging favorite movie embeddings
        inferred_user_embedding = np.mean(favorite_embeddings, axis=0)

      
        genre_preferences = {}
        total_rated_movies = len(valid_movie_ids)

        # collect genres from favorite movies and their importance
        valid_genres = set()
        for movie_id in valid_movie_ids:
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                for genre in self.genre_names:
                    if movie_row[genre].values[0] == 1:
                        # Weight the genre by the movie's similarity to other favorites
                        genre_preferences[genre] = genre_preferences.get(genre, 0) + (1.0 / total_rated_movies)
                        valid_genres.add(genre)

        # normalize genre preferences
        if genre_preferences:
            total = sum(genre_preferences.values())
            for genre in genre_preferences:
                genre_preferences[genre] /= total

    
        def calculate_movie_similarity(favorite_embeddings, movie_embedding):
            # Calculate embedding similarity
            similarities = [np.dot(fav_emb, movie_embedding) / (
                np.linalg.norm(fav_emb) * np.linalg.norm(movie_embedding) + 1e-8
            ) for fav_emb in favorite_embeddings]
            
            # use max similarity 
            return max(similarities)

        # compute similarity scores and recommendations
        all_movies = []

        for idx in range(len(movie_embeddings)):
            movie_id = self.inv_movie_id_map.get(idx)
            if movie_id is None or movie_id in favorite_movie_ids:  
                continue
                
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if movie_row.empty:
                continue
            
    
            movie_emb = movie_embeddings[idx]
            
            # compute similarity between user and movie embeddings
            similarity = calculate_movie_similarity(favorite_embeddings, movie_emb)
            
   
            movie_genres = [g for g in self.genre_names if movie_row[g].values[0] == 1]
            genre_str = "|".join(movie_genres)
            
          
            if not set(movie_genres) & valid_genres:
                continue  
            
            # calculate genre preference score
            genre_score = sum(genre_preferences.get(g, 0) for g in movie_genres)
            
            # combine embedding similarity with genre preferences
            combined_score = 0.7 * similarity + 0.3 * genre_score
            
            # calculate predicted rating (scaled to original rating range)
            pred_rating = ((combined_score + 1) / 2) * (self.rating_max - self.rating_min) + self.rating_min
            
            # cap rating within bounds
            pred_rating = max(min(pred_rating, self.rating_max), self.rating_min)
            
            # store movie with its prediction
            popularity = movie_row['movie_rating_count'].values[0]
            all_movies.append((movie_id, movie_row['title'].values[0], pred_rating, genre_str, movie_emb, popularity))

        # if not enough movies found, fallback to popularity-based recommendations
        if len(all_movies) < top_n:
            print(f"Only {len(all_movies)} movies found matching genre preferences. Falling back to popular movies.")
           
            genre_movies = self.movies_df[
                self.movies_df[list(valid_genres)].sum(axis=1) > 0
            ].sort_values('movie_rating_count', ascending=False).head(top_n)
            
            fallback_movies = [
                (row['movieId'], row['title'], 0.0, 
                    "|".join([g for g in self.genre_names if row[g] == 1]), 
                    movie_embeddings[self.movie_id_map[row['movieId']]], 
                    row['movie_rating_count'])
                for _, row in genre_movies.iterrows()
            ]
            all_movies.extend(fallback_movies)

        # Apply diversity-aware recommendation selection (MMR algorithm)
        lambda_param = 1.0 - diversity_factor
        selected = []
        candidate_pool = all_movies.copy()

        def cosine_similarity(vec1, vec2):
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot / (norm1 * norm2 + 1e-8)

        # normalize relevance scores 
        relevances = [p[2] for p in candidate_pool]
        rel_min, rel_max = min(relevances), max(relevances) if relevances else (0, 5)
        relevance_range = rel_max - rel_min if rel_max != rel_min else 1

        # max popularity for normalization
        max_popularity = max([p[5] for p in candidate_pool]) if candidate_pool else 1

        # threshold for uniqueness
        similarity_threshold = 0.95 if len(candidate_pool) > top_n * 2 else 0.98

        # keep selecting until we have enough recommendations or run out of candidates
        while len(selected) < top_n and candidate_pool:
            mmr_scores = []
            for candidate in candidate_pool:
                movie_id, title, relevance, genres_str, emb_vec, popularity = candidate
                
                # normalize relevance to [0,1]
                norm_relevance = (relevance - rel_min) / relevance_range
                
                # calculate diversity penalty
                if not selected:
                    # first item has no diversity penalty
                    diversity_penalty = 0
                else:
                    # calculate similarities to already selected items
                    similarities = [cosine_similarity(emb_vec, s[4]) for s in selected]
                    
                    # skip if too similar to already selected items
                    if max(similarities) >= similarity_threshold:
                        continue
                        
                    # use average similarity as diversity penalty
                    similarity_boost = np.mean(similarities)
                    diversity_penalty = similarity_boost ** 2
                
                # calculate popularity penalty
                popularity_penalty = np.log1p(popularity) / np.log1p(max_popularity) * 0.2
                
                # calculate final MMR score
                mmr_score = lambda_param * norm_relevance - (1 - lambda_param) * diversity_penalty - popularity_penalty
                
                mmr_scores.append((mmr_score, candidate))
            
            # if no valid candidates after similarity filtering, adjust threshold or break
            if not mmr_scores:
                if len(selected) < top_n / 2:
                    similarity_threshold += 0.05  # Make threshold more lenient
                    continue
                else:
                    break
            
            # sort by MMR score and select the best one
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_item = mmr_scores[0][1]
            
            # add to selected items and remove from candidate pool
            selected.append(best_item)
            candidate_pool.remove(best_item)

        # format the output
        recommendations = [(title, rating, genres) for _, title, rating, genres, _, _ in selected]

        # ensure we return exactly top_n recommendations or all available if fewer
        return recommendations[:top_n]

    
    def load_data(self):
        """Load and preprocess the MovieLens dataset with advanced features."""
        print("Loading data...")
        
        # load ratings data
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        self.ratings_df = pd.read_csv(ratings_path)
        
        # load movies data
        movies_path = os.path.join(self.data_path, 'movies.csv')
        self.movies_df = pd.read_csv(movies_path)
        
        # extract year from movie title and create a feature
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$', expand=False)
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        # fill missing years with median
        median_year = self.movies_df['year'].median()
        self.movies_df['year'] = self.movies_df['year'].fillna(median_year)
        
        # extract genres and create one-hot encoding
        self.movies_df['genres'] = self.movies_df['genres'].str.split('|')
        genres_list = self.movies_df['genres'].explode().unique().tolist()
        if '(no genres listed)' in genres_list:
            genres_list.remove('(no genres listed)')
        self.genre_names = sorted(genres_list)
        self.n_genres = len(self.genre_names)
        
        # create genre features
        for genre in self.genre_names:
            self.movies_df[genre] = self.movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
        
        # calculate movie popularity as number of ratings
        movie_popularity = self.ratings_df.groupby('movieId').size().reset_index(name='movie_rating_count')
        self.movies_df = pd.merge(self.movies_df, movie_popularity, on='movieId', how='left')
        self.movies_df['movie_rating_count'] = self.movies_df['movie_rating_count'].fillna(0)
        
        # calculate movie average rating
        movie_avg_rating = self.ratings_df.groupby('movieId')['rating'].mean().reset_index(name='movie_avg_rating')
        self.movies_df = pd.merge(self.movies_df, movie_avg_rating, on='movieId', how='left')
        self.movies_df['movie_avg_rating'] = self.movies_df['movie_avg_rating'].fillna(self.ratings_df['rating'].mean())
        
        # calculate user activity as number of individual ratings
        user_activity = self.ratings_df.groupby('userId').size().reset_index(name='user_rating_count')
        self.ratings_df = pd.merge(self.ratings_df, user_activity, on='userId', how='left')
        
        # calculate user average rating 
        user_avg_rating = self.ratings_df.groupby('userId')['rating'].mean().reset_index(name='user_avg_rating')
        self.ratings_df = pd.merge(self.ratings_df, user_avg_rating, on='userId', how='left')
        
        # add movie release year to ratings
        self.ratings_df = pd.merge(self.ratings_df, 
                                  self.movies_df[['movieId', 'year', 'movie_rating_count', 'movie_avg_rating']], 
                                  on='movieId', how='left')
        
        # add timestamp features
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        self.ratings_df['day_of_week'] = self.ratings_df['timestamp'].dt.dayofweek
        self.ratings_df['hour_of_day'] = self.ratings_df['timestamp'].dt.hour
        self.ratings_df['month'] = self.ratings_df['timestamp'].dt.month
        self.ratings_df['year_rated'] = self.ratings_df['timestamp'].dt.year
        
        # calculate how recent a rating is
        max_timestamp = self.ratings_df['timestamp'].max()
        self.ratings_df['days_since_rated'] = (max_timestamp - self.ratings_df['timestamp']).dt.total_seconds() / (24 * 60 * 60)
        
        # calculate movie age at rating time
        self.ratings_df['movie_age'] = self.ratings_df['year_rated'] - self.ratings_df['year']
        
        # get number of unique users and movies
        self.n_users = self.ratings_df['userId'].nunique()
        self.n_movies = self.ratings_df['movieId'].nunique()
        
        # store min and max ratings for scaling
        self.rating_min = self.ratings_df['rating'].min()
        self.rating_max = self.ratings_df['rating'].max()
        
        print(f"Loaded {len(self.ratings_df)} ratings from {self.n_users} users on {self.n_movies} movies")
        print(f"Found {self.n_genres} unique genres")
        
        # create a mapping for user and movie ids
        user_ids = self.ratings_df['userId'].unique()
        movie_ids = self.ratings_df['movieId'].unique()
        
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
        self.movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
        
        # create inverse mappings for prediction phase
        self.inv_user_id_map = {v: k for k, v in self.user_id_map.items()}
        self.inv_movie_id_map = {v: k for k, v in self.movie_id_map.items()}
        
        # map IDs in the dataset
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_id_map)
        self.ratings_df['movie_idx'] = self.ratings_df['movieId'].map(self.movie_id_map)
        
        # merge ratings with movie features
        self.data = pd.merge(self.ratings_df, self.movies_df[['movieId'] + self.genre_names], on='movieId')
        
        return self.ratings_df, self.movies_df
    
    def prepare_train_val_test_split(self, val_size=0.1, test_size=0.1, stratify_recent=True):
        """Split the data into training, validation, and test sets with enhanced stratification."""
        print("Preparing train/validation/test splits...")
        
        # create feature matrix
        X = self.ratings_df[['user_idx', 'movie_idx']].values
        
        # create user and movie popularity features
        user_popularity = self.ratings_df[['user_rating_count']].values
        movie_popularity = self.ratings_df[['movie_rating_count']].values
        user_avg = self.ratings_df[['user_avg_rating']].values
        movie_avg = self.ratings_df[['movie_avg_rating']].values
        
        # scale popularity features
        self.user_popularity_scaler = MinMaxScaler()
        self.movie_popularity_scaler = MinMaxScaler()
        user_popularity_scaled = self.user_popularity_scaler.fit_transform(user_popularity)
        movie_popularity_scaled = self.movie_popularity_scaler.fit_transform(movie_popularity)
        
        # normalize the average ratings to [0,1] similar to target
        user_avg_scaled = (user_avg - self.rating_min) / (self.rating_max - self.rating_min)
        movie_avg_scaled = (movie_avg - self.rating_min) / (self.rating_max - self.rating_min)
        

        time_features = self.ratings_df[['day_of_week', 'hour_of_day', 'month', 'days_since_rated', 'movie_age']].values
        self.time_scaler = StandardScaler()
        time_features_scaled = self.time_scaler.fit_transform(time_features)
        

        y = self.ratings_df['rating'].values
        
        # get genre features for each movie
        genre_features = np.zeros((len(self.ratings_df), self.n_genres))
        for i, row in enumerate(self.ratings_df.itertuples()):
            movie_id = row.movieId
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                genre_features[i] = movie_row[self.genre_names].values[0]
        
        # to ensure each split has both recent and older ratings
        if stratify_recent:
            # create a binary variable for recent ratings 
            recent_threshold = np.percentile(self.ratings_df['days_since_rated'], 25)  # Last 25% of ratings
            is_recent = (self.ratings_df['days_since_rated'] <= recent_threshold).astype(int)
            
            # split ensuring both recent and older ratings are distributed
            X_train, X_temp, y_train, y_temp, genre_train, genre_temp, time_train, time_temp, \
            user_pop_train, user_pop_temp, movie_pop_train, movie_pop_temp, \
            user_avg_train, user_avg_temp, movie_avg_train, movie_avg_temp = train_test_split(
                X, y, genre_features, time_features_scaled, 
                user_popularity_scaled, movie_popularity_scaled,
                user_avg_scaled, movie_avg_scaled,
                test_size=val_size + test_size, stratify=is_recent, random_state=42)
            
            # split temp into validation and test sets
           
            X_val, X_test, y_val, y_test, genre_val, genre_test, time_val, time_test, \
            user_pop_val, user_pop_test, movie_pop_val, movie_pop_test, \
            user_avg_val, user_avg_test, movie_avg_val, movie_avg_test = train_test_split(
                X_temp, y_temp, genre_temp, time_temp, 
                user_pop_temp, movie_pop_temp,
                user_avg_temp, movie_avg_temp,
                test_size=0.2, random_state=42)
        else:
            # regular split without stratification
            X_train, X_temp, y_train, y_temp, genre_train, genre_temp, time_train, time_temp, \
            user_pop_train, user_pop_temp, movie_pop_train, movie_pop_temp, \
            user_avg_train, user_avg_temp, movie_avg_train, movie_avg_temp = train_test_split(
                X, y, genre_features, time_features_scaled, 
                user_popularity_scaled, movie_popularity_scaled,
                user_avg_scaled, movie_avg_scaled,
                test_size=val_size + test_size, random_state=42)
            
            # split temp into validation and test sets
            X_val, X_test, y_val, y_test, genre_val, genre_test, time_val, time_test, \
            user_pop_val, user_pop_test, movie_pop_val, movie_pop_test, \
            user_avg_val, user_avg_test, movie_avg_val, movie_avg_test = train_test_split(
                X_temp, y_temp, genre_temp, time_temp, 
                user_pop_temp, movie_pop_temp,
                user_avg_temp, movie_avg_temp,
                test_size=0.2, random_state=42)
        
        # store the splits with additional features
        self.train_data = (X_train, y_train, genre_train, time_train, 
                         user_pop_train, movie_pop_train, user_avg_train, movie_avg_train)
        self.val_data = (X_val, y_val, genre_val, time_val, 
                       user_pop_val, movie_pop_val, user_avg_val, movie_avg_val)
        self.test_data = (X_test, y_test, genre_test, time_test, 
                        user_pop_test, movie_pop_test, user_avg_test, movie_avg_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return self.train_data, self.val_data, self.test_data
    
    def build_model(self):
    
        #  model builder function from the separate module
        self.model = build_recommender_model(
            n_users=self.n_users,
            n_movies=self.n_movies,
            n_genres=self.n_genres,
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            use_bias=self.use_bias,
            use_features=self.use_features,
            use_time_features=self.use_time_features
        )
        
        return self.model
    
    def train(self, epochs=100, patience=10):
        if self.use_features:
            X_train, y_train, genre_train, time_train, user_pop_train, movie_pop_train, user_avg_train, movie_avg_train = self.train_data
            X_val, y_val, genre_val, time_val, user_pop_val, movie_pop_val, user_avg_val, movie_avg_val = self.val_data
        else:
            X_train, y_train = self.train_data[0], self.train_data[1]
            X_val, y_val = self.val_data[0], self.val_data[1]
        
        
        y_train_scaled = (y_train - self.rating_min) / (self.rating_max - self.rating_min)
        y_val_scaled = (y_val - self.rating_min) / (self.rating_max - self.rating_min)
        
      
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=0.00001, 
                verbose=1
            ),
            ModelCheckpoint(
                'best_improved_model.keras', 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            )
        ]
        
        # prepare inputs based on model type
        if self.use_features:
            train_inputs = [
                X_train[:, 0],               # user_idx
                X_train[:, 1],               # movie_idx
                genre_train,                 # genre features
                user_pop_train,              # user popularity
                movie_pop_train,             # movie popularity
                user_avg_train,              # user average rating
                movie_avg_train              # movie average rating
            ]
            
            val_inputs = [
                X_val[:, 0],                 # user_idx
                X_val[:, 1],                 # movie_idx
                genre_val,                   # genre features
                user_pop_val,                # user popularity
                movie_pop_val,               # movie popularity
                user_avg_val,                # user average rating
                movie_avg_val                # movie average rating
            ]
            
           
            if self.use_time_features:
                train_inputs.append(time_train)
                val_inputs.append(time_val)
        else:
            train_inputs = [X_train[:, 0], X_train[:, 1]]
            val_inputs = [X_val[:, 0], X_val[:, 1]]
        
        # train model
        self.history = self.model.fit(
            train_inputs, y_train_scaled,
            validation_data=(val_inputs, y_val_scaled),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1, shuffle=False
        )
        
        return self.history
    
    def evaluate(self):
       
        if self.use_features:
            X_test, y_test, genre_test, time_test, user_pop_test, movie_pop_test, user_avg_test, movie_avg_test = self.test_data
        else:
            X_test, y_test = self.test_data[0], self.test_data[1]
        
        if self.use_features:
            test_inputs = [
                X_test[:, 0],               # user_idx
                X_test[:, 1],               # movie_idx
                genre_test,                 # genre features
                user_pop_test,              # user popularity
                movie_pop_test,             # movie popularity
                user_avg_test,              # user average rating
                movie_avg_test              # movie average rating
            ]
            
            # add time features if used
            if self.use_time_features:
                test_inputs.append(time_test)
        else:
            test_inputs = [X_test[:, 0], X_test[:, 1]]
        
        # get predictions
        y_pred_scaled = self.model.predict(test_inputs)
        
        # rescale predictions to original rating scale
        y_pred = y_pred_scaled * (self.rating_max - self.rating_min) + self.rating_min
        
        # calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        self.best_rmse = min(self.best_rmse, rmse)
        self.best_mae = min(self.best_mae, mae)

        errors = y_test - y_pred.flatten()
        
        # plot evaluation metrics
        plot_evaluation_metrics(y_test, y_pred.flatten(), errors, rmse, mae)
        
        # if features are used, perform detailed error analysis
        if self.use_features:
            detailed_error_analysis(
                X_test, y_test, y_pred.flatten(), errors,
                user_pop_test, movie_pop_test
            )
        
        # calculate Precision and Recall@5
        precision_at_5, recall_at_5 = self.calculate_precision_recall_at_k(X_test, y_test, k=5)
        print(f"Precision@5: {precision_at_5:.4f}")
        print(f"Recall@5: {recall_at_5:.4f}")
        
        return rmse, mae, precision_at_5, recall_at_5
    
    def calculate_movie_similarity(favorite_embeddings, movie_embedding):
        # calculate embedding similarity
        similarities = [np.dot(fav_emb, movie_embedding) / (
            np.linalg.norm(fav_emb) * np.linalg.norm(movie_embedding) + 1e-8
        ) for fav_emb in favorite_embeddings]
        
        # use the max similarity as it indicates the closest match
        return max(similarities)
    
    def _plot_ranking_metrics(self, results, k_values):
       
        plt.figure(figsize=(10, 6))
        
        precisions = [results[k]['precision'] for k in k_values]
        recalls = [results[k]['recall'] for k in k_values]
        f1_scores = [results[k]['f1'] for k in k_values]
        
        plt.plot(k_values, precisions, 'b-', marker='o', label='Precision')
        plt.plot(k_values, recalls, 'g-', marker='s', label='Recall')
        plt.plot(k_values, f1_scores, 'r-', marker='^', label='F1 Score')
        
        plt.xlabel('k value')
        plt.ylabel('Score')
        plt.title('Ranking Metrics at different k values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(k_values)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('ranking_metrics.png')
        plt.close()

    def calculate_precision_recall_at_k(self, X_test, y_test, k=5, threshold=3.5):
        """
        calculates precision and recall@k for the test set.
        
        Args:
            X_test: Test set features
            y_test: Test set ratings
            k: The 'k' in Precision@k and Recall@k
            threshold: Rating threshold for relevance 
            
        Returns:
            precision_at_k, recall_at_k
        """
        # group test data by user
        user_ids = X_test[:, 0]
        movie_ids = X_test[:, 1]
        
        # dict to store user-movie pairs
        user_items = {}
        user_relevant_items = {}
        
        # group by user and identify relevant items 
        for i, user_id in enumerate(user_ids):
            if user_id not in user_items:
                user_items[user_id] = []
                user_relevant_items[user_id] = []
            
            user_items[user_id].append((movie_ids[i], y_test[i]))
            if y_test[i] >= threshold:
                user_relevant_items[user_id].append(movie_ids[i])
        
        # calculate precision and recall for each user
        precision_sum = 0
        recall_sum = 0
        user_count = 0
        
        for user_id in user_items:
            # skip users with no relevant items
            if not user_relevant_items[user_id]:
                continue
            
            # sort items by actual rating in descending order and take top k
            sorted_items = sorted(user_items[user_id], key=lambda x: x[1], reverse=True)
            top_k_items = [item[0] for item in sorted_items[:k]]
            
            # count relevant items in top k recommendations
            hit_count = len(set(top_k_items) & set(user_relevant_items[user_id]))
            
            # calculate precision and recall for this user
            precision = hit_count / k if k > 0 else 0
            recall = hit_count / len(user_relevant_items[user_id]) if user_relevant_items[user_id] else 0
            
            precision_sum += precision
            recall_sum += recall
            user_count += 1
        
        # calculate avg precision and recall
        avg_precision = precision_sum / user_count if user_count > 0 else 0
        avg_recall = recall_sum / user_count if user_count > 0 else 0
        
        return avg_precision, avg_recall

    def evaluate_ranking_metrics(self, k_values=[1, 5, 10], threshold=3.5):
        """
        evaluate model using various ranking metrics at different k values.
        
        Args:
            k_values: List of k values to calculate metrics at
            threshold: Rating threshold for relevance
            
        Returns:
            Dictionary with metrics for each k value
        """
        if self.use_features:
            X_test, y_test = self.test_data[0], self.test_data[1]
        else:
            X_test, y_test = self.test_data[0], self.test_data[1]
        
        results = {}
        
        for k in k_values:
            precision, recall = self.calculate_precision_recall_at_k(X_test, y_test, k=k, threshold=threshold)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[k] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"At k={k}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        
        # plot metrics at different k values
        self._plot_ranking_metrics(results, k_values)
        
        return results
    def plot_training_history(self):
        """plot the training and validation loss curves with improved visualization."""
        if self.history is None:
            print("no training history available. Train model first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Huber)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # plot MAE
        plt.subplot(2, 1, 2)
        plt.plot(self.history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(self.history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png')
    
    def get_recommendations(self, user_id, top_n=24, diversity_factor=0.4, include_seen=False):

        try:
            if user_id not in self.user_id_map:
                print(f"[INFO] user_id {user_id} not in model, using fallback index 0")
                internal_user_index = 0
            else:
                internal_user_index = self.user_id_map[user_id]

            user_inputs = np.array([internal_user_index] * self.n_movies)
            movie_inputs = np.arange(self.n_movies)

            if self.use_features:
                genres = self.movies_df[self.genre_names].values[:self.n_movies]
                user_popularity = np.zeros((self.n_movies, 1))
                user_avg = np.zeros((self.n_movies, 1))
                movie_popularity = self.movies_df['movie_rating_count'].values[:self.n_movies].reshape(-1, 1)
                movie_avg = self.movies_df['movie_avg_rating'].values[:self.n_movies].reshape(-1, 1)

                user_pop_scaled = self.user_popularity_scaler.transform(user_popularity) if self.user_popularity_scaler else user_popularity
                movie_pop_scaled = self.movie_popularity_scaler.transform(movie_popularity) if self.movie_popularity_scaler else movie_popularity
                user_avg_scaled = self.user_popularity_scaler.transform(user_avg) if self.user_popularity_scaler else user_avg
                movie_avg_scaled = self.movie_popularity_scaler.transform(movie_avg) if self.movie_popularity_scaler else movie_avg

                all_inputs = [
                    user_inputs, movie_inputs, genres,
                    user_pop_scaled, movie_pop_scaled,
                    user_avg_scaled, movie_avg_scaled
                ]

                if self.use_time_features:
                    avg_time_features = np.zeros((self.n_movies, 5))
                    avg_time_features = self.time_scaler.transform(avg_time_features) if self.time_scaler else avg_time_features
                    all_inputs.append(avg_time_features)

                predictions = self.model.predict(all_inputs, verbose=0)
            else:
                predictions = self.model.predict([user_inputs, movie_inputs], verbose=0)

            predictions = predictions * (self.rating_max - self.rating_min) + self.rating_min

            movie_embedding_layer = self.model.get_layer('movie_embedding')
            movie_embeddings = movie_embedding_layer.get_weights()[0][:self.n_movies]

            movie_predictions = []
            for idx, pred in zip(range(self.n_movies), predictions):
                movie_id = self.inv_movie_id_map[idx]
                movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
                if not movie_row.empty:
                    title = movie_row['title'].values[0]
                    genres_str = "|".join([g for g in self.genre_names if movie_row[g].values[0] == 1])
                    embedding_vec = movie_embeddings[idx]
                    popularity = movie_row['movie_rating_count'].values[0]
                    movie_predictions.append((movie_id, title, pred[0], genres_str, embedding_vec, popularity))

            if not include_seen:
                rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
                movie_predictions = [p for p in movie_predictions if p[0] not in rated_movies]

            # ensure we have enough movies to recommend
            if len(movie_predictions) < top_n:
                # if not enough movies after filtering, include some popular ones
                # that the user hasn't seen yet 
                print(f"Not enough movies available after filtering ({len(movie_predictions)}), including popular ones.")
                popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n * 2)
                
                for _, row in popular_movies.iterrows():
                    movie_id = row['movieId']
                    # check if this movie is already in our predictions
                    if movie_id in [p[0] for p in movie_predictions]:
                        continue
                        
                    # check if user has already seen this movie
                    if not include_seen and movie_id in rated_movies:
                        continue
                        
                    # get the movie embedding
                    if movie_id in self.movie_id_map:
                        movie_idx = self.movie_id_map[movie_id]
                        embedding_vec = movie_embeddings[movie_idx] if movie_idx < len(movie_embeddings) else np.zeros(self.embedding_dim)
                    else:
                        continue  
                        
                    # add to predictions with a slightly lower score than the lowest current prediction
                    min_pred = min([p[2] for p in movie_predictions]) if movie_predictions else 3.0
                    title = row['title']
                    genres_str = "|".join([g for g in self.genre_names if row[g] == 1])
                    popularity = row['movie_rating_count']
                    
                    movie_predictions.append((movie_id, title, min_pred - 0.1, genres_str, embedding_vec, popularity))
                    
                    # break if we have enough predictions 
                    if len(movie_predictions) >= top_n * 1.5:  # get 50% more than needed to allow for diversity
                        break

            # proceed with Maximal Marginal Relevance (MMR) selection for diversity
            lambda_param = 1.0 - diversity_factor  
            selected = []
            candidate_pool = movie_predictions.copy()

            def cosine_similarity(vec1, vec2):
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                return dot / (norm1 * norm2 + 1e-8)

            # normalize relevance scores to [0,1] for consistent weight
            relevances = [p[2] for p in candidate_pool]
            rel_min, rel_max = min(relevances), max(relevances) if relevances else (0, 5)  
            relevance_range = rel_max - rel_min if rel_max != rel_min else 1

            # max popularity for normalization
            max_popularity = max([p[5] for p in candidate_pool]) if candidate_pool else 1

            # lower threshold for uniqueness when more recommendations are needed
            similarity_threshold = 0.95 if len(candidate_pool) > top_n * 2 else 0.98

            # keep selecting until we have enough recommendations or run out of candidates
            while len(selected) < top_n and candidate_pool:
                mmr_scores = []
                for candidate in candidate_pool:
                    movie_id, title, relevance, genres_str, emb_vec, popularity = candidate
                    
                    # normalize relevance to [0,1]
                    norm_relevance = (relevance - rel_min) / relevance_range
                    
                    # calculate diversity penalty 
                    if not selected:
                        # first item has no diversity penalty
                        diversity_penalty = 0
                    else:
                        # calculate similarities to already selected items
                        similarities = [cosine_similarity(emb_vec, s[4]) for s in selected]
                        
                        # skip this candidate if it's too similar to something already selected
                        if max(similarities) >= similarity_threshold:
                            continue
                            
                        # use average similarity as diversity penalty
                        similarity_boost = np.mean(similarities)
                        diversity_penalty = similarity_boost ** 2  # Square to penalize high similarity more
                    
                    # Calculate popularity penalty 
                    popularity_penalty = np.log1p(popularity) / np.log1p(max_popularity) * 0.2
                    
                    # calculate final MMR score
                    mmr_score = lambda_param * norm_relevance - (1 - lambda_param) * diversity_penalty - popularity_penalty
                    
                    mmr_scores.append((mmr_score, candidate))
                
                # if no valid candidates after similarity filtering, adjust treshold or break
                if not mmr_scores:
                    # if we have less than half of requested recommendations, 
                    # lower similarity threshold and try again with remaining candidates
                    if len(selected) < top_n / 2:
                        similarity_threshold += 0.05  
                        continue
                    else:
                        break
                
                # sort by MMR score and select the best one
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                best_item = mmr_scores[0][1]
                
                # add to selected items and remove from candidate pool
                selected.append(best_item)
                candidate_pool.remove(best_item)

            # format the output
            recommendations = [(title, rating, genres) for _, title, rating, genres, _, _ in selected]
            
            # ensure we return exactly top_n recommendations or all available if fewer
            return recommendations[:top_n]

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            
            # fallback to popular movies
            try:
                popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
                return [(row['title'], 0, "") for _, row in popular_movies.iterrows()]
            except:
                # ultimate fallback if everything fails
                return [("Movie 1", 0, ""), ("Movie 2", 0, ""), ("Movie 3", 0, "")]

    def get_user_genre_preferences(self, user_id):
        """Calculate user genre preferences based on their rating history."""
        # get all ratings by this user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            return {}
        
        # get movies rated by the user
        rated_movies = user_ratings['movieId'].unique()
        
        # get genre information for these movies
        genre_data = self.movies_df[self.movies_df['movieId'].isin(rated_movies)]
        
        if genre_data.empty:
            return {}
        
        # calculate weighted average rating by genre
        genre_ratings = {}
        for genre in self.genre_names:
            # get movies with this genre that the user rated
            genre_movies = genre_data[genre_data[genre] == 1]['movieId']
            
            if len(genre_movies) == 0:
                continue
                
            # get user ratings for these movies
            ratings_for_genre = user_ratings[user_ratings['movieId'].isin(genre_movies)]['rating']
            
            if len(ratings_for_genre) > 0:
                # calculate average rating for this genre
                avg_rating = ratings_for_genre.mean()
                # Store average rating weighted by number of movies rated in this genre
                genre_ratings[genre] = avg_rating * len(ratings_for_genre)
        
        return genre_ratings
    # save model
    def save_model(self, filepath="saved_model"):
        
        if self.model is None:
            print("No trained model to save.")
            return
            
        # create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # save the Keras model
        self.model.save(os.path.join(filepath, "model.keras"))
        
        # save metadata
        metadata = {
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "n_genres": self.n_genres,
            "genre_names": self.genre_names,
            "embedding_dim": self.embedding_dim,
            "use_features": self.use_features,
            "use_time_features": self.use_time_features,
            "use_bias": self.use_bias,
            "rating_min": float(self.rating_min),
            "rating_max": float(self.rating_max),
            "best_rmse": float(self.best_rmse),
            "best_mae": float(self.best_mae),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # save the mappings
        np.save(os.path.join(filepath, "user_id_map.npy"), self.user_id_map)
        np.save(os.path.join(filepath, "movie_id_map.npy"), self.movie_id_map)
        np.save(os.path.join(filepath, "metadata.npy"), metadata)
        
        # save the scalers
        if self.time_scaler is not None:
            with open(os.path.join(filepath, "time_scaler.pkl"), 'wb') as f:
                pickle.dump(self.time_scaler, f)
        if self.user_popularity_scaler is not None:
            with open(os.path.join(filepath, "user_popularity_scaler.pkl"), 'wb') as f:
                pickle.dump(self.user_popularity_scaler, f)
        if self.movie_popularity_scaler is not None:
            with open(os.path.join(filepath, "movie_popularity_scaler.pkl"), 'wb') as f:
                pickle.dump(self.movie_popularity_scaler, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="saved_model"):
            """
            Load a saved model and its metadata.
            
            Args:
                filepath: path to the directory containing the saved model files
                
            Returns:
                bool: True if model was successfully loaded
            """
            try:
                if not os.path.exists(filepath):
                    print(f"Model directory {filepath} not found")
                    return False
                
                # load metadata
                try:
                    metadata = np.load(os.path.join(filepath, "metadata.npy"), allow_pickle=True).item()
                except:
                    print(f"Failed to load metadata from {filepath}")
                    return False
                
                # restore model properties from metadata
                self.n_users = metadata.get('n_users')
                self.n_movies = metadata.get('n_movies')
                self.n_genres = metadata.get('n_genres')
                self.genre_names = metadata.get('genre_names')
                self.rating_min = metadata.get('rating_min')
                self.rating_max = metadata.get('rating_max')
                self.best_rmse = metadata.get('best_rmse', float('inf'))
                self.best_mae = metadata.get('best_mae', float('inf'))
                
                # load ID mappings
                try:
                    self.user_id_map = np.load(os.path.join(filepath, "user_id_map.npy"), allow_pickle=True).item()
                    self.movie_id_map = np.load(os.path.join(filepath, "movie_id_map.npy"), allow_pickle=True).item()
                    # create inverse mappings
                    self.inv_user_id_map = {v: k for k, v in self.user_id_map.items()}
                    self.inv_movie_id_map = {v: k for k, v in self.movie_id_map.items()}
                except:
                    print(f"Failed to load ID mappings from {filepath}")
                    return False
                
                # load the scalers
                try:
                    if os.path.exists(os.path.join(filepath, "time_scaler.pkl")):
                        with open(os.path.join(filepath, "time_scaler.pkl"), 'rb') as f:
                            self.time_scaler = pickle.load(f)
                            
                    if os.path.exists(os.path.join(filepath, "user_popularity_scaler.pkl")):
                        with open(os.path.join(filepath, "user_popularity_scaler.pkl"), 'rb') as f:
                            self.user_popularity_scaler = pickle.load(f)
                            
                    if os.path.exists(os.path.join(filepath, "movie_popularity_scaler.pkl")):
                        with open(os.path.join(filepath, "movie_popularity_scaler.pkl"), 'rb') as f:
                            self.movie_popularity_scaler = pickle.load(f)
                except:
                    print("couldn't load scalers")
                
                # load the Keras model with custom objects for Lambda layers
                try:
                    self.model = tf.keras.models.load_model(
                        os.path.join(filepath, "model.keras"), 
                        custom_objects=CUSTOM_OBJECTS,
                        compile=True
                    )
                except Exception as e:
                    print(f"Failed to load model from {filepath}: {e}")
                    # try again with different loading method if first attempt fails
                    try:
                        print("Attempting alternative loading method...")
                        self.model = tf.keras.models.load_model(
                            os.path.join(filepath, "model.keras"),
                            custom_objects=CUSTOM_OBJECTS,
                            compile=False
                        )
                        # recompile model
                        self.model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                            loss=tf.keras.losses.Huber(delta=1.0),
                            metrics=['mean_absolute_error']
                        )
                        print("Successfully loaded model with alternative method.")
                    except Exception as e2:
                        print(f"All loading attempts failed: {e2}")
                        return False
                
                print(f"Model loaded successfully from {filepath}")
                print(f"Model metadata: {metadata.get('n_users')} users, {metadata.get('n_movies')} movies")
                print(f"Best performance: RMSE={metadata.get('best_rmse', 'N/A')}, MAE={metadata.get('best_mae', 'N/A')}")
                
                return True
            except Exception as e:
                print(f"Unexpected error loading model: {e}")
                import traceback
                traceback.print_exc()
                return False