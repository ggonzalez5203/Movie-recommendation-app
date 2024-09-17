# recommendation_system.py

# Step 1: Import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Step 2: Load and process the data
def load_data():
    # Load the user ratings data
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load the movie titles data
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=['movie_id', 'title'], usecols=[0, 1], encoding='latin-1')
    
    # Merge ratings and movies data
    data = pd.merge(ratings, movies, on='movie_id')
    
    return data

# Step 3: Create user-movie matrix and calculate similarity
def create_similarity_matrix(data):
    user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    
    # Compute cosine similarity between movies
    movie_similarity = cosine_similarity(user_movie_matrix_filled.T)
    
    return pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Step 4: Recommendation function
def recommend_movies(movie_title, similarity_matrix, top_n=5):
    if movie_title not in similarity_matrix.columns:
        return f"Movie '{movie_title}' not found."
    
    similar_movies = similarity_matrix[movie_title].sort_values(ascending=False)
    similar_movies = similar_movies[similar_movies.index != movie_title]
    
    return similar_movies.head(top_n)

# Step 5: Test the recommendation system
if __name__ == '__main__':
    data = load_data()
    movie_similarity_df = create_similarity_matrix(data)
    
    # Example: Recommend 5 movies similar to "Star Wars (1977)"
    recommendations = recommend_movies('Star Wars (1977)', movie_similarity_df)
    print(recommendations)
