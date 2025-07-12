"""
neural network model architecture for the recommender system.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, Multiply
from tensorflow.keras.layers import BatchNormalization, Add, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
import os
import json

def zeros_like_function(x):
    return tf.zeros_like(x)

def zeros_like_output_shape(input_shape):
    return input_shape

def scale_by_factor(x, factor=0.1):
    return x * factor

def scale_by_factor_output_shape(input_shape):
    return input_shape

def scale_by_0_1(x):
    return scale_by_factor(x, 0.1)


# try loading optimal params from file, or just use some default params
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(base_dir, "..", "optimal_model_results", "model_params.json")
    
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            optimal_params = json.load(f)
    else:
        optimal_params = {
            "embedding_dim": 128,
            "learning_rate": 0.0005,
            "dropout_rate": 0.3,
            "l2_reg": 0.00005
        }
except Exception as e:
    print(f"warning: could not load optimal parameters: {e}")
    optimal_params = {
        "embedding_dim": 128,
        "learning_rate": 0.0005,
        "dropout_rate": 0.3,
        "l2_reg": 0.00005
    }


def build_recommender_model(
    n_users, 
    n_movies, 
    n_genres, 
    embedding_dim=None, 
    learning_rate=None, 
    dropout_rate=None, 
    l2_reg=None, 
    use_bias=True, 
    use_features=True, 
    use_time_features=True
):
    """
    builds the recommender model using embeddings and dense layers.
    has optional genre and time features.
    """
    # if values not given, fall back 
    embedding_dim = embedding_dim if embedding_dim is not None else optimal_params["embedding_dim"]
    learning_rate = learning_rate if learning_rate is not None else optimal_params["learning_rate"]
    dropout_rate = dropout_rate if dropout_rate is not None else optimal_params["dropout_rate"]
    l2_reg = l2_reg if l2_reg is not None else optimal_params["l2_reg"]
    
    # input layers for user and movie
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    
    # embedding layers for users and movies
    user_embedding = Embedding(n_users, embedding_dim, embeddings_regularizer=l2(l2_reg), name='user_embedding')(user_input)
    movie_embedding = Embedding(n_movies, embedding_dim, embeddings_regularizer=l2(l2_reg), name='movie_embedding')(movie_input)
    
    # flatten the embedding layers
    user_flatten = Flatten()(user_embedding)
    movie_flatten = Flatten()(movie_embedding)
    
    # bias terms for users and movies 
    if use_bias:
        user_bias = Embedding(n_users, 1, embeddings_initializer='zeros',
                              embeddings_regularizer=l2(l2_reg), name='user_bias')(user_input)
        movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros',
                               embeddings_regularizer=l2(l2_reg), name='movie_bias')(movie_input)
        user_bias = Flatten()(user_bias)
        movie_bias = Flatten()(movie_bias)
        
       
        global_bias = Lambda(zeros_like_function, 
                             output_shape=zeros_like_output_shape, 
                             name='global_bias')(user_bias)
    
   
    if use_features:
        genre_input = Input(shape=(n_genres,), name='genre_input')
        user_popularity_input = Input(shape=(1,), name='user_popularity_input')
        movie_popularity_input = Input(shape=(1,), name='movie_popularity_input')
        user_avg_input = Input(shape=(1,), name='user_avg_input')
        movie_avg_input = Input(shape=(1,), name='movie_avg_input')
        
        all_inputs = [user_input, movie_input, genre_input, 
                      user_popularity_input, movie_popularity_input,
                      user_avg_input, movie_avg_input]
        
        # add time input if that flag is set
        if use_time_features:
            time_input = Input(shape=(5,), name='time_input')  
            all_inputs.append(time_input)
            time_dense = Dense(16, activation='relu', kernel_initializer=GlorotUniform(seed=42))(time_input)
            time_features = BatchNormalization()(time_dense)
        
      
        genre_dense = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), kernel_initializer=GlorotUniform(seed=42))(genre_input)
        genre_bn = BatchNormalization()(genre_dense)
        
        # multiply user and movie vectors 
        mf_vector = Multiply()([user_flatten, movie_flatten])
        
        # put everything together into one big feature vector
        if use_time_features:
            feature_vector = Concatenate()([
                mf_vector, genre_bn, time_features,
                user_popularity_input, movie_popularity_input,
                user_avg_input, movie_avg_input
            ])
        else:
            feature_vector = Concatenate()([
                mf_vector, genre_bn,
                user_popularity_input, movie_popularity_input,
                user_avg_input, movie_avg_input
            ])
        
        # first dense block
        dense1 = Dense(256, kernel_regularizer=l2(l2_reg), kernel_initializer=GlorotUniform(seed=42))(feature_vector)
        bn1 = BatchNormalization()(dense1)
        act1 = LeakyReLU(alpha=0.1)(bn1)
        drop1 = Dropout(dropout_rate, seed=42)(act1)
        
        # second dense block with residual connection
        dense2 = Dense(128, kernel_regularizer=l2(l2_reg), kernel_initializer=GlorotUniform(seed=42))(drop1)
        bn2 = BatchNormalization()(dense2)
        act2 = LeakyReLU(alpha=0.1)(bn2)
        drop2 = Dropout(dropout_rate,seed=42)(act2)
        
        # make projection so can add it to previous block 
        projection = Dense(128)(drop1)
        resid1 = Add()([drop2, projection])
        
        
        dense3 = Dense(64, kernel_regularizer=l2(l2_reg), kernel_initializer=GlorotUniform(seed=42))(resid1)
        bn3 = BatchNormalization()(dense3)
        act3 = LeakyReLU(alpha=0.1)(bn3)
        drop3 = Dropout(dropout_rate)(act3)
        
        # final prediction
        if use_bias:
            output_pre = Dense(1, activation='sigmoid', name='prediction_pre', kernel_initializer=GlorotUniform(seed=42))(drop3)
            biases = Add()([user_bias, movie_bias, global_bias])
            scaled_biases = Lambda(scale_by_0_1, output_shape=scale_by_factor_output_shape, name='scaled_biases')(biases)
            output = Add(name='prediction')([output_pre, scaled_biases])
        else:
            output = Dense(1, activation='sigmoid', name='prediction', kernel_initializer=GlorotUniform(seed=42))(drop3)
        
        model = Model(inputs=all_inputs, outputs=output)
    
    else:
        # no features just going with pure matrix factorization
        dot_product = Multiply()([user_flatten, movie_flatten])
        dense = Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42))(dot_product)
        
        if use_bias:
            output_pre = Dense(1, activation='sigmoid', name='prediction_pre', kernel_initializer=GlorotUniform(seed=42))(dense)
            biases = Add()([user_bias, movie_bias, global_bias])
            scaled_biases = Lambda(scale_by_0_1, output_shape=scale_by_factor_output_shape, name='scaled_biases')(biases)
            output = Add(name='prediction')([output_pre, scaled_biases])
        else:
            output = Dense(1, activation='sigmoid', name='prediction', kernel_initializer=GlorotUniform(seed=42))(dense)
        
        model = Model(inputs=[user_input, movie_input], outputs=output)
    
    # compile the model using huber loss 
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mean_absolute_error']
    )
    
    model.summary()
    return model
