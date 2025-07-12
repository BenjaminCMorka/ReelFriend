"""
explanation code for the recommender, uses SHAP 
to explain why each movie got recommended
"""
import numpy as np

def print_recommendation_explanation(recommender, user_id, recommendations):
    """
    generate user-friendly explanations for why each movie was recommended
    """
    # check if this is new user with no training data
    is_new_user = user_id not in recommender.user_id_map
    
    if is_new_user:
        return _explain_recommendations_for_new_user(recommender, user_id, recommendations)
    else:
        return _explain_recommendations_for_existing_user(recommender, user_id, recommendations)

def _explain_recommendations_for_new_user(recommender, user_id, recommendations):
    """
    explanation logic for new users based on fav genres and similarities
    """
    favorite_movie_ids = []
    
    # grab their ratings if they exist
    user_ratings = recommender.ratings_df[recommender.ratings_df["userId"] == user_id]
    if not user_ratings.empty:
        favorites = user_ratings[user_ratings["rating"] >= 4.0]
        favorite_movie_ids = favorites["movieId"].tolist()
    
    # gather genres and titles of those favorites
    favorite_genres = {}
    favorite_titles = []
    
    for movie_id in favorite_movie_ids:
        movie_row = recommender.movies_df[recommender.movies_df["movieId"] == movie_id]
        if movie_row.empty:
            continue
            
        title = movie_row["title"].values[0]
        favorite_titles.append(title)
        
        for genre in recommender.genre_names:
            if movie_row[genre].values[0] == 1:
                favorite_genres[genre] = favorite_genres.get(genre, 0) + 1
    
    # get top genres they like
    favorite_genres = sorted(favorite_genres.items(), key=lambda x: x[1], reverse=True)
    
    explanations = []
    print("\nRecommendations with natural language explanations:")
    
    for title, pred_rating, genres_str in recommendations:
        rec_row = recommender.movies_df[recommender.movies_df["title"] == title]
        if rec_row.empty:
            explanation = f"I think you'll enjoy {title} based on your taste."
            print(explanation)
            explanations.append(explanation)
            continue
        
        movie_genres = [g for g in recommender.genre_names if rec_row[g].values[0] == 1]
        common_genres = set(movie_genres) & set([g for g, _ in favorite_genres])
        
        # figure out the most similar favorite movie based on genres
        most_similar = None
        max_overlap = 0
        for fav_title in favorite_titles:
            fav_row = recommender.movies_df[recommender.movies_df["title"] == fav_title]
            if fav_row.empty:
                continue
            fav_genres = [g for g in recommender.genre_names if fav_row[g].values[0] == 1]
            overlap = len(set(movie_genres) & set(fav_genres))
            if overlap > max_overlap:
                max_overlap = overlap
                most_similar = fav_title
        
        reasons = []
        if common_genres:
            genres_list = list(common_genres)[:2]
            if len(genres_list) == 1 and genres_list[0][0].lower() in 'aeiou':
                genre_reason = f"it's an {genres_list[0]} movie"
            else:
                genre_reason = f"it's a {', '.join(genres_list)} movie"
            if len(common_genres) > 2:
                genre_reason += " (genres you enjoy)"
            reasons.append(genre_reason)
        
        if rec_row["movie_rating_count"].values[0] > recommender.movies_df["movie_rating_count"].median():
            reasons.append("it's popular among viewers")
        
        if rec_row["movie_avg_rating"].values[0] > 4.0:
            reasons.append("it has high ratings from other users")
        
        if reasons:
            explanation = f"- {title} (Predicted Rating: {pred_rating:.2f})\n  Recommended because {', and '.join(reasons)}"
            if most_similar:
                explanation += f", similar to \"{most_similar}\" which you liked."
            else:
                explanation += "."
        else:
            explanation = f"- {title} (Predicted Rating: {pred_rating:.2f})\n  I think you'll enjoy this based on your overall taste in movies."
        
        print(explanation)
        explanations.append(explanation)
    
    return explanations

def _explain_recommendations_for_existing_user(recommender, user_id, recommendations):
    """
    generate explanations for a known user using their rating history
    """
    explanations = []
    genre_preferences = {}

    user_ratings = recommender.ratings_df[recommender.ratings_df["userId"] == user_id]
    
    if not user_ratings.empty:
        liked_movies = user_ratings[user_ratings["rating"] >= 4.0]
        for _, row in liked_movies.iterrows():
            movie_id = row["movieId"]
            movie_row = recommender.movies_df[recommender.movies_df["movieId"] == movie_id]
            if not movie_row.empty:
                for genre in recommender.genre_names:
                    if movie_row[genre].values[0] == 1:
                        genre_preferences[genre] = genre_preferences.get(genre, 0) + 1
    
    if genre_preferences:
        total = sum(genre_preferences.values())
        for genre in genre_preferences:
            genre_preferences[genre] /= total
    
    watched_movies = set(user_ratings["movieId"].unique())
    
    print("\nRecommendations with explanations:")
    
    for title, pred_rating, genres_str in recommendations:
        rec_row = recommender.movies_df[recommender.movies_df["title"] == title]
        if rec_row.empty:
            explanation = f"I think you'll enjoy {title} based on your taste."
            print(explanation)
            explanations.append(explanation)
            continue
        
        movie_id = rec_row["movieId"].values[0]
        movie_genres = [g for g in recommender.genre_names if rec_row[g].values[0] == 1]
        genre_score = sum(genre_preferences.get(g, 0) for g in movie_genres)
        
        most_similar = None
        max_similarity = 0
        liked_movie_ids = user_ratings[user_ratings["rating"] >= 4.0]["movieId"].unique()
        
        for liked_id in liked_movie_ids:
            liked_row = recommender.movies_df[recommender.movies_df["movieId"] == liked_id]
            if liked_row.empty:
                continue
            liked_genres = [g for g in recommender.genre_names if liked_row[g].values[0] == 1]
            intersection = len(set(movie_genres) & set(liked_genres))
            union = len(set(movie_genres) | set(liked_genres))
            similarity = intersection / union if union > 0 else 0
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = liked_row["title"].values[0]
        
        reasons = []
        top_genres = sorted([(g, genre_preferences.get(g, 0)) for g in movie_genres], 
                            key=lambda x: x[1], reverse=True)[:2]
        
        if top_genres and top_genres[0][1] > 0:
            if len(top_genres) == 1 and top_genres[0][0][0].lower() in 'aeiou':
                genre_reason = f"it's an {top_genres[0][0]} movie"
            else:
                genre_reason = f"it's a {', '.join([g[0] for g in top_genres])} movie"
            if len(movie_genres) > 2:
                genre_reason += " (genres you enjoy)"
            reasons.append(genre_reason)
        
        if rec_row["movie_rating_count"].values[0] > recommender.movies_df["movie_rating_count"].median():
            reasons.append("it's popular among viewers")
        
        if rec_row["movie_avg_rating"].values[0] > 4.0:
            reasons.append("it has high ratings from other users")
        
        if movie_id not in watched_movies:
            reasons.append("it's different from what you've watched before")
        
        if reasons:
            explanation = f"- {title} (Predicted Rating: {pred_rating:.2f})\n  Recommended because {', and '.join(reasons)}"
            if most_similar and max_similarity > 0.5:
                explanation += f", similar to \"{most_similar}\" which you liked."
            else:
                explanation += "."
        else:
            explanation = f"- {title} (Predicted Rating: {pred_rating:.2f})\n  I think you'll enjoy this based on your viewing history and rating patterns."
        
        print(explanation)
        explanations.append(explanation)
    
    return explanations
