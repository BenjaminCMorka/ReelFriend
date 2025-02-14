import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import logging
import gc  # Garbage collector

logging.basicConfig(level=logging.INFO)

def calculate_chunked_similarity():
    # Load the processed dataset
    logging.info("Loading dataset...")
    movies = pd.read_csv('data/movies_processed.csv')
    
    # Get genre columns
    genre_columns = [col for col in movies.columns 
                    if col not in ['movieId', 'title', 'genres', 'year']]
    
    # Convert to sparse matrix
    logging.info("Converting to sparse matrix...")
    movie_features_sparse = csr_matrix(movies[genre_columns].values)
    
    # Get dimensions
    n_movies = movie_features_sparse.shape[0]
    chunk_size = 1000  # Adjust this based on your available RAM
    
    # Initialize an empty array to store the results
    similarity_matrix = np.memmap('data/similarity_matrix.mmap', 
                                dtype='float32', 
                                mode='w+',
                                shape=(n_movies, n_movies))
    
    logging.info(f"Processing {n_movies} movies in chunks of {chunk_size}")
    
    # Process in chunks
    for i in range(0, n_movies, chunk_size):
        chunk_end = min(i + chunk_size, n_movies)
        logging.info(f"Processing movies {i} to {chunk_end}")
        
        # Calculate similarity for this chunk
        chunk_similarities = cosine_similarity(
            movie_features_sparse[i:chunk_end], 
            movie_features_sparse
        )
        
        # Store the chunk
        similarity_matrix[i:chunk_end] = chunk_similarities
        
        # Force garbage collection
        del chunk_similarities
        gc.collect()
        
    
        # Save progress periodically
        similarity_matrix.flush()
    
    # Convert memmap to regular numpy array and save
    logging.info("Saving final matrix...")
    final_matrix = np.array(similarity_matrix)
    np.save('data/cosine_similarity_matrix.npy', final_matrix)
    
    # Clean up memmap file
    del similarity_matrix
    import os
    os.remove('data/similarity_matrix.mmap')
    
    logging.info("Process completed successfully!")

if __name__ == "__main__":
    calculate_chunked_similarity()