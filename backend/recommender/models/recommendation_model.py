import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from data_processing.data_loader import load_movielens_data
from data_processing.feature_engineering import prepare_train_val_test_split
from models.model_builder import build_recommender_model

# random seeds 
np.random.seed(42)
tf.random.set_seed(42)


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
            use_features: Whether to use movie features (genres) in the model
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
        

        self.best_rmse = float('inf')
        self.best_mae = float('inf')
        

        self.time_scaler = None
        self.user_popularity_scaler = None
        self.movie_popularity_scaler = None

    def update_user_embedding(self, user_id, new_rating):
        """
        Update a specific user's embedding vector without retraining the full model.
        This allows for incremental updates as users rate new movies.
        
        Args:
            user_id: ID of the user whose embedding needs to be updated
            new_rating: Dict containing movie_id and rating for the new rating
        
        Returns:
            Boolean indicating success of the update
        """
        try:
            print(f"Updating embedding for user {user_id} with new rating: {new_rating}")
            
            # check if user exists in the model
            if user_id not in self.user_id_map:
                print(f"[WARN] User {user_id} not found in model embeddings.")
                return False
            
            # check if movie exists in model
            movie_id = int(new_rating['movie_id'])
            if movie_id not in self.movie_id_map:
                print(f"[WARN] Movie {movie_id} not found in model.")
                return False
                
            # get the user's internal index in model
            user_idx = self.user_id_map[user_id]
            
            # extract user embedding layer from  model
            user_embedding_layer = self.model.get_layer('user_embedding')
            user_embeddings = user_embedding_layer.get_weights()[0]
            
            # get the current user embedding
            current_embedding = user_embeddings[user_idx].copy()
            
            # get the movie's internal index
            movie_idx = self.movie_id_map[movie_id]
            
            # get movie embedding
            movie_embedding_layer = self.model.get_layer('movie_embedding')
            movie_embeddings = movie_embedding_layer.get_weights()[0]
            movie_embedding = movie_embeddings[movie_idx]
            
            # normalize rating to [0,1] range
            normalized_rating = (float(new_rating['rating']) - self.rating_min) / (self.rating_max - self.rating_min)
            
            # update embedding using weighted avg
            updated_embedding = 0.9 * current_embedding + 0.1 * normalized_rating * movie_embedding
            
            # update the user's embedding in weights
            user_embeddings[user_idx] = updated_embedding
            
            # set the updated weights back to  layer
            user_embedding_layer.set_weights([user_embeddings])
            
            print(f"[INFO] Successfully updated embedding for user {user_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update user embedding: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_new_user_ratings(self, user_id, movie_ratings):
        """
        Add new user ratings to the existing ratings dataset
        
        Args:
            user_id: ID of the user
            movie_ratings: List of tuples (movie_id, rating)
        """
        # convert string movie ids to int
        movie_ratings = [(int(movie_id), float(rating)) for movie_id, rating in movie_ratings]
        
        # create new ratings for specific user
        new_user_ratings = []
        for movie_id, rating in movie_ratings:
            new_user_ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating,
                'timestamp': int(datetime.now().timestamp())
            })
        
        # convert to DataFrame
        new_ratings_df = pd.DataFrame(new_user_ratings)
        
        # remove any existing ratings for user
        self.ratings_df = self.ratings_df[self.ratings_df['userId'] != user_id]
        
        # concatenate new ratings
        self.ratings_df = pd.concat([self.ratings_df, new_ratings_df], ignore_index=True)
        
        # reload data to update mappings
        self.load_data()
        
        return self

    def recommend_for_user(self, user_id, favorite_movie_ids=None, top_n=24, diversity_factor=0.4):
        """
        Generate recommendations for a user, handling both existing and new users.
        For new users, estimates user latent factors based on their favorite movies.
        
        Args:
            user_id: ID of the user 
            favorite_movie_ids: List of movie IDs the user likes for cold start
            top_n: Number of recommendations to generate
            diversity_factor: Balance between accuracy and diversity (0-1)
            
        Returns:
            List of (title, predicted_rating, genres) tuples
        """
      
        is_new_user = user_id not in self.user_id_map
        
        if is_new_user and (favorite_movie_ids is None or len(favorite_movie_ids) == 0):
            print(f"new user {user_id} without favorite movies. using popular recs.")
            # fall back to popularitybased recommendations
            popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
            return [(row['title'], 0.0, "|".join([g for g in self.genre_names if row[g] == 1])) 
                    for _, row in popular_movies.iterrows()]
        
        if is_new_user:
            print(f"New user {user_id} with {len(favorite_movie_ids)} favorite movies. Inferring preferences.")
            # get embeddings for their favorite movies and infer user latent factors
            return self._recommend_for_new_user(user_id, favorite_movie_ids, top_n, diversity_factor)
        else:
            # if existing user use regular recommendation logic
            return self.get_recommendations(user_id, top_n, diversity_factor)
        
    def _recommend_for_new_user(self, user_id, favorite_movie_ids, top_n=24, diversity_factor=0.4):
        """
        Generate recommendations for a new user based on their favorite movies.
        
        Args:
            user_id: ID for the new user
            favorite_movie_ids: List of movie IDs the user likes
            top_n: Number of recommendations to generate
            diversity_factor: Balance between accuracy and diversity (0-1)
            
        Returns:
            List of (title, predicted_rating, genres) tuples
        """
        # map favorite movie ids to internal indices
        valid_movie_ids = []
        valid_indices = []
        
        for movie_id in favorite_movie_ids:
            if movie_id in self.movie_id_map:
                valid_movie_ids.append(movie_id)
                valid_indices.append(self.movie_id_map[movie_id])
        
        if not valid_indices:
            print("None of the provided favorite movies were found in the model.")
            # fallback to popularity based recs
            popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
            return [(row['title'], 0.0, "|".join([g for g in self.genre_names if row[g] == 1])) 
                    for _, row in popular_movies.iterrows()]
        
        # get movie embeddings layer from model
        movie_embedding_layer = self.model.get_layer('movie_embedding')
        movie_embeddings = movie_embedding_layer.get_weights()[0]
        
        # get embeddings for user's favorite movies
        favorite_embeddings = movie_embeddings[valid_indices]
        
        # infer the user's latent factors by averaging favorite movie embeddings
        inferred_user_embedding = np.mean(favorite_embeddings, axis=0)
        
        # get genre preferences from favorite movies
        genre_preferences = {}
        for movie_id in valid_movie_ids:
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                for genre in self.genre_names:
                    if movie_row[genre].values[0] == 1:
                        genre_preferences[genre] = genre_preferences.get(genre, 0) + 1
        
        # normalize genre preferences
        if genre_preferences:
            total = sum(genre_preferences.values())
            for genre in genre_preferences:
                genre_preferences[genre] /= total
        
        # compute similarity scores between inferred user embedding
        # and all movie embeddings to find best matches
        all_movies = []
        
        for idx in range(len(movie_embeddings)):
            movie_id = self.inv_movie_id_map.get(idx)
            if movie_id is None or movie_id in favorite_movie_ids:  # Skip favorites - already liked
                continue
                
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if movie_row.empty:
                continue
                
            # compute cosine similarity between user and movie embedding
            movie_emb = movie_embeddings[idx]
            similarity = np.dot(inferred_user_embedding, movie_emb) / (
                np.linalg.norm(inferred_user_embedding) * np.linalg.norm(movie_emb) + 1e-8)
            

            title = movie_row['title'].values[0]
            
            # get genre overlap with preferences
            movie_genres = [g for g in self.genre_names if movie_row[g].values[0] == 1]
            genre_str = "|".join(movie_genres)
            
            # calculate a genre preference score
            genre_score = sum(genre_preferences.get(g, 0) for g in movie_genres)
            
            # combine embedding similarity with genre preferences with weight
            combined_score = 0.7 * similarity + 0.3 * genre_score
            
            # calculate predicted rating and scale to original rating range
            pred_rating = ((combined_score + 1) / 2) * (self.rating_max - self.rating_min) + self.rating_min
            
            # cap rating within bounds
            pred_rating = max(min(pred_rating, self.rating_max), self.rating_min)
            
            # store movie with its prediction
            popularity = movie_row['movie_rating_count'].values[0]
            all_movies.append((movie_id, title, pred_rating, genre_str, movie_emb, popularity))
        
        # apply diversity recommendation selection MMR algorithm
      
        
        # initialize variables for MMR selection
        lambda_param = 1.0 - diversity_factor  
        selected = []
        candidate_pool = all_movies.copy()
        
        def cosine_similarity(vec1, vec2):
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot / (norm1 * norm2 + 1e-8)
        
        # normalize relevance scores for consistent weighting
        relevances = [p[2] for p in candidate_pool]
        rel_min, rel_max = min(relevances), max(relevances) if relevances else (0, 5)
        relevance_range = rel_max - rel_min if rel_max != rel_min else 1
        
        # max popularity for normalization
        max_popularity = max([p[5] for p in candidate_pool]) if candidate_pool else 1
        
        # threshold for uniqueness
        similarity_threshold = 0.95 if len(candidate_pool) > top_n * 2 else 0.98
        
        # keep selecting until  enough recs or no more candidates
        while len(selected) < top_n and candidate_pool:
            mmr_scores = []
            for candidate in candidate_pool:
                movie_id, title, relevance, genres_str, emb_vec, popularity = candidate
                
                # normalize relevance to [0,1]
                norm_relevance = (relevance - rel_min) / relevance_range
                
                # calculate diversity penalty
                if not selected:
                 
                    diversity_penalty = 0
                else:
                    # calc similarities to already selected items
                    similarities = [cosine_similarity(emb_vec, s[4]) for s in selected]
                    
                    # skip if too similar 
                    if max(similarities) >= similarity_threshold:
                        continue
                        
                    # use avg similarity as diversity penalty
                    similarity_boost = np.mean(similarities)
                    diversity_penalty = similarity_boost ** 2
                
               
                popularity_penalty = np.log1p(popularity) / np.log1p(max_popularity) * 0.2
                
                mmr_score = lambda_param * norm_relevance - (1 - lambda_param) * diversity_penalty - popularity_penalty
                
                mmr_scores.append((mmr_score, candidate))
            
            # if no valid candidates after similarity filter change threshold or break
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
        
        # use the imported data loader function
        self.ratings_df, self.movies_df = load_movielens_data(self.data_path)
        
        # extract genre names from movies_df
        genres_list = self.movies_df['genres'].explode().unique().tolist()
        if '(no genres listed)' in genres_list:
            genres_list.remove('(no genres listed)')
        self.genre_names = sorted(genres_list)
        self.n_genres = len(self.genre_names)
        
        # store min and max ratings for scaling
        self.rating_min = self.ratings_df['rating'].min()
        self.rating_max = self.ratings_df['rating'].max()
        
        # get number of unique users and movies
        self.n_users = self.ratings_df['userId'].nunique()
        self.n_movies = self.ratings_df['movieId'].nunique()
        
        # create a mapping for user and movie ids
        user_ids = self.ratings_df['userId'].unique()
        movie_ids = self.ratings_df['movieId'].unique()
        
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
        self.movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
        
        # create inverse mappings for prediction phase
        self.inv_user_id_map = {v: k for k, v in self.user_id_map.items()}
        self.inv_movie_id_map = {v: k for k, v in self.movie_id_map.items()}
        
        # map ids in the dataset
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_id_map)
        self.ratings_df['movie_idx'] = self.ratings_df['movieId'].map(self.movie_id_map)
        
        # merge ratings with movie features
        self.data = pd.merge(self.ratings_df, self.movies_df[['movieId'] + self.genre_names], on='movieId')
        
        return self.ratings_df, self.movies_df
    
    def prepare_train_val_test_split(self, val_size=0.1, test_size=0.1, stratify_recent=True):
     
    
        self.train_data, self.val_data, self.test_data, \
        self.user_popularity_scaler, self.movie_popularity_scaler, self.time_scaler = \
            prepare_train_val_test_split(
                self.ratings_df, self.movies_df, self.n_genres, self.genre_names,
                self.rating_min, self.rating_max, val_size, test_size, stratify_recent
            )
        
        return self.train_data, self.val_data, self.test_data
    
    def build_model(self):
    

       
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
       
        print(f"training model for up to {epochs} epochs...")
        
        # unpack training and validation data
        if self.use_features:
            X_train, y_train, genre_train, time_train, user_pop_train, movie_pop_train, user_avg_train, movie_avg_train = self.train_data
            X_val, y_val, genre_val, time_val, user_pop_val, movie_pop_val, user_avg_val, movie_avg_val = self.val_data
        else:
            X_train, y_train = self.train_data[0], self.train_data[1]
            X_val, y_val = self.val_data[0], self.val_data[1]
        
        # scale ratings to [0, 1]
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
        
        # prepare inputs 
        if self.use_features:
            train_inputs = [
                X_train[:, 0],               # user_idx
                X_train[:, 1],               # movie_idx
                genre_train,                 # genre features
                user_pop_train,              # user popularity
                movie_pop_train,             # movie popularity
                user_avg_train,              # user avg rating
                movie_avg_train              # movie avg rating
            ]
            
            val_inputs = [
                X_val[:, 0],                 # user_idx
                X_val[:, 1],                 # movie_idx
                genre_val,                   # genre features
                user_pop_val,                # user popularity
                movie_pop_val,               # movie popularity
                user_avg_val,                # user avg rating
                movie_avg_val                # movie avg rating
            ]
            
           
            if self.use_time_features:
                train_inputs.append(time_train)
                val_inputs.append(time_val)
        else:
            train_inputs = [X_train[:, 0], X_train[:, 1]]
            val_inputs = [X_val[:, 0], X_val[:, 1]]
        
        # train  model
        self.history = self.model.fit(
            train_inputs, y_train_scaled,
            validation_data=(val_inputs, y_val_scaled),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self):

        
        # unpack test data
        if self.use_features:
            X_test, y_test, genre_test, time_test, user_pop_test, movie_pop_test, user_avg_test, movie_avg_test = self.test_data
        else:
            X_test, y_test = self.test_data[0], self.test_data[1]
        
        # scale ratings to [0, 1]
        y_test_scaled = (y_test - self.rating_min) / (self.rating_max - self.rating_min)
        
        # prepare inputs based on model type
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
            
            #add time features if used
            if self.use_time_features:
                test_inputs.append(time_test)
        else:
            test_inputs = [X_test[:, 0], X_test[:, 1]]
        
        # get predictions
        y_pred_scaled = self.model.predict(test_inputs)
        
        # rescale predictions to original rating scale
        y_pred = y_pred_scaled * (self.rating_max - self.rating_min) + self.rating_min
        

        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        self.best_rmse = min(self.best_rmse, rmse)
        self.best_mae = min(self.best_mae, mae)
        
        # error analysis
        errors = y_test - y_pred.flatten()
        

        self.plot_evaluation_metrics(y_test, y_pred.flatten(), errors, rmse, mae)
        
        
        if self.use_features:
            self.detailed_error_analysis(
                X_test, y_test, y_pred.flatten(), errors,
                user_pop_test, movie_pop_test
            )
        
        return rmse, mae
    
    def plot_evaluation_metrics(self, y_true, y_pred, errors, rmse, mae):
       
        plt.figure(figsize=(15, 10))
        
        # rrror distribution
        plt.subplot(2, 2, 1)
        plt.hist(errors, bins=30, alpha=0.7, color='blue')
        plt.title(f'Error Distribution (RMSE={rmse:.4f}, MAE={mae:.4f})')
        plt.xlabel('Error (True - Predicted)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # actual vs pred
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.1, color='blue')
        plt.plot([0, 5], [0, 5], 'r--')
        plt.title('Actual vs Predicted Ratings')
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        plt.axis('equal')
        plt.axis([0.5, 5.5, 0.5, 5.5])
        plt.grid(True, alpha=0.3)
        
        # error vs actual rating
        plt.subplot(2, 2, 3)
        plt.scatter(y_true, errors, alpha=0.1, color='blue')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Error vs Actual Rating')
        plt.xlabel('Actual Rating')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        
        # error vs predicted rating
        plt.subplot(2, 2, 4)
        plt.scatter(y_pred, errors, alpha=0.1, color='blue')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Error vs Predicted Rating')
        plt.xlabel('Predicted Rating')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
    
    def detailed_error_analysis(self, X_test, y_true, y_pred, errors, user_pop, movie_pop):
       
        plt.figure(figsize=(15, 10))
        
        # error by user Activity
        plt.subplot(2, 2, 1)
        user_activity = user_pop.flatten()
        plt.scatter(user_activity, np.abs(errors), alpha=0.1)
        plt.title('Absolute Error vs User Activity')
        plt.xlabel('User Rating Count (normalized)')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        # error by movie popularity
        plt.subplot(2, 2, 2)
        movie_popularity = movie_pop.flatten()
        plt.scatter(movie_popularity, np.abs(errors), alpha=0.1)
        plt.title('Absolute Error vs Movie Popularity')
        plt.xlabel('Movie Rating Count (normalized)')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        # error Distribution by Rating
        plt.subplot(2, 2, 3)
        rating_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        rating_groups = pd.cut(y_true, rating_bins, labels=['1', '2', '3', '4', '5'])
        error_by_rating = pd.DataFrame({'error': errors, 'rating_group': rating_groups})
        error_by_rating.boxplot(column='error', by='rating_group', ax=plt.gca())
        plt.title('Error Distribution by Rating')
        plt.xlabel('True Rating')
        plt.ylabel('Error')
        plt.suptitle('')
        
        # RMSE by rating
        plt.subplot(2, 2, 4)
        rmse_by_rating = error_by_rating.groupby('rating_group')['error'].apply(lambda x: np.sqrt(np.mean(x**2)))
        rmse_by_rating.plot(kind='bar', ax=plt.gca())
        plt.title('RMSE by Rating')
        plt.xlabel('True Rating')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png')
    
    def plot_training_history(self):
       
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Huber)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot MAE
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
        """
        Generate top-N movie recommendations for a given user using MMR for diversity,
        combining embedding diversity with popularity suppression.
        """
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

            # ensure enough movies to recommend
            if len(movie_predictions) < top_n:
                # if not have enough movies after filtering, include popular 
                popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n * 2)
                
                for _, row in popular_movies.iterrows():
                    movie_id = row['movieId']
                   
                    if movie_id in [p[0] for p in movie_predictions]:
                        continue
                        
                    # check if user has already seen  movie
                    if not include_seen and movie_id in rated_movies:
                        continue
                        
                    # get the movie embedding
                    if movie_id in self.movie_id_map:
                        movie_idx = self.movie_id_map[movie_id]
                        embedding_vec = movie_embeddings[movie_idx] if movie_idx < len(movie_embeddings) else np.zeros(self.embedding_dim)
                    else:
                        continue  # skip if movie_id not in map
                        
                    # add to predictions with lower score than the lowest current prediction
                    min_pred = min([p[2] for p in movie_predictions]) if movie_predictions else 3.0
                    title = row['title']
                    genres_str = "|".join([g for g in self.genre_names if row[g] == 1])
                    popularity = row['movie_rating_count']
                    
                    movie_predictions.append((movie_id, title, min_pred - 0.1, genres_str, embedding_vec, popularity))
                    
                    # break if we have enough predictions now
                    if len(movie_predictions) >= top_n * 1.5: 
                        break

            # proceed with maximal marginal relevance selection for diversity
            lambda_param = 1.0 - diversity_factor  
            selected = []
            candidate_pool = movie_predictions.copy()

            def cosine_similarity(vec1, vec2):
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                return dot / (norm1 * norm2 + 1e-8)

            # normalize relevance scores to [0,1] for consistent weighting
            relevances = [p[2] for p in candidate_pool]
            rel_min, rel_max = min(relevances), max(relevances) if relevances else (0, 5)  # Use rating scale bounds if empty
            relevance_range = rel_max - rel_min if rel_max != rel_min else 1

         
            max_popularity = max([p[5] for p in candidate_pool]) if candidate_pool else 1

            # lower threshold for uniqueness when more recs needed
            similarity_threshold = 0.95 if len(candidate_pool) > top_n * 2 else 0.98

           
            while len(selected) < top_n and candidate_pool:
                mmr_scores = []
                for candidate in candidate_pool:
                    movie_id, title, relevance, genres_str, emb_vec, popularity = candidate
                    
                    
                    norm_relevance = (relevance - rel_min) / relevance_range
                    
                    
                    if not selected:
                       
                        diversity_penalty = 0
                    else:
                       
                        similarities = [cosine_similarity(emb_vec, s[4]) for s in selected]
                        
                       
                        if max(similarities) >= similarity_threshold:
                            continue
                            
                       
                        similarity_boost = np.mean(similarities)
                        diversity_penalty = similarity_boost ** 2  
                    
                    #
                    popularity_penalty = np.log1p(popularity) / np.log1p(max_popularity) * 0.2
                    
                    
                    mmr_score = lambda_param * norm_relevance - (1 - lambda_param) * diversity_penalty - popularity_penalty
                    
                    mmr_scores.append((mmr_score, candidate))
                
                
                if not mmr_scores:
                  
                    if len(selected) < top_n / 2:
                        similarity_threshold += 0.05  
                        continue
                    else:
                        break
                
                # sort by MMR score and  select best 
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                best_item = mmr_scores[0][1]
                
                # add to selected items and remove from candidate pool
                selected.append(best_item)
                candidate_pool.remove(best_item)

            
            recommendations = [(title, rating, genres) for _, title, rating, genres, _, _ in selected]
            
            return recommendations[:top_n]

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            
            #
            try:
                popular_movies = self.movies_df.sort_values('movie_rating_count', ascending=False).head(top_n)
                return [(row['title'], 0, "") for _, row in popular_movies.iterrows()]
            except:
               
                return [("Movie 1", 0, ""), ("Movie 2", 0, ""), ("Movie 3", 0, "")]

    def get_user_genre_preferences(self, user_id):
        # get all ratings by  user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            return {}
        
        # get movies rated by the user
        rated_movies = user_ratings['movieId'].unique()
        
        # get genre information for these movies
        genre_data = self.movies_df[self.movies_df['movieId'].isin(rated_movies)]
        
        if genre_data.empty:
            return {}
        
        # calculate weighted avg rating by genre
        genre_ratings = {}
        for genre in self.genre_names:
            # get movies with this genre that the user rated
            genre_movies = genre_data[genre_data[genre] == 1]['movieId']
            
            if len(genre_movies) == 0:
                continue
                
            # get user ratings for  movies
            ratings_for_genre = user_ratings[user_ratings['movieId'].isin(genre_movies)]['rating']
            
            if len(ratings_for_genre) > 0:
                # calculate average rating for this genre
                avg_rating = ratings_for_genre.mean()
                # store avg rating weighted by number of movies rated in this genre
                genre_ratings[genre] = avg_rating * len(ratings_for_genre)
        
        return genre_ratings
    
    def save_model(self, filepath="saved_model"):
        
        if self.model is None:
            print("No trained model to save.")
            return
            
       
        os.makedirs(filepath, exist_ok=True)
        
        # save keras model
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
        
        # save mappings
        np.save(os.path.join(filepath, "user_id_map.npy"), self.user_id_map)
        np.save(os.path.join(filepath, "movie_id_map.npy"), self.movie_id_map)
        np.save(os.path.join(filepath, "metadata.npy"), metadata)
        
        # save scalers
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
            Load previously saved recommender model and its metadata.
            
            Args:
                filepath: Path to the directory containing the saved model files
                
            Returns:
                bool: True if model was successfully loaded
            """
            from models import model_builder
            
            
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
                
                # load id mappings
                try:
                    self.user_id_map = np.load(os.path.join(filepath, "user_id_map.npy"), allow_pickle=True).item()
                    self.movie_id_map = np.load(os.path.join(filepath, "movie_id_map.npy"), allow_pickle=True).item()
                    # create inverse mappings
                    self.inv_user_id_map = {v: k for k, v in self.user_id_map.items()}
                    self.inv_movie_id_map = {v: k for k, v in self.movie_id_map.items()}
                except:
                    print(f"Failed to load ID mappings from {filepath}")
                    return False
                
                # load  scalers
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
                    print("Warning: Failed to load some scalers. This may affect prediction quality.")
                
                # load Keras model with custom objects for lambda layers
                try:
                    self.model = tf.keras.models.load_model(
                        os.path.join(filepath, "model.keras"), 
                        custom_objects={'model_builder': model_builder},
                        compile=True, safe_mode=False
                    )
                except Exception as e:
                    print(f"Failed to load model from {filepath}: {e}")
                   
                    try:
                       
                        self.model = tf.keras.models.load_model(
                            os.path.join(filepath, "model.keras"),
                            custom_objects={'model_builder': model_builder},
                            compile=False
                        )
                        # recompile 
                        self.model.compile(
                            optimizer=Adam(learning_rate=self.learning_rate),
                            loss=tf.keras.losses.Huber(delta=1.0),
                            metrics=['mean_absolute_error']
                        )
                       
                    except Exception as e2:
                        print(f"loading failed {e2}")
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