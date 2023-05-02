# !pip install surprise
import pandas as pd
from typing import List
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import seaborn as sns

import matplotlib.pyplot as plt
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

"""
Import Data
"""

# movies = pd.read_csv('movies.csv')
movies = pd.DataFrame()

for i in range(2):
    file_path = f'movies_{i}.csv'
    split = pd.read_csv(file_path)
    movies = pd.concat([movies, split], ignore_index=True)

ratings = pd.read_csv('ratings.csv')
links = pd.read_csv('links.csv')
keywords = pd.read_csv('keywords.csv')

credits = pd.DataFrame()

for i in range(11):
    file_path = f'credits_{i}.csv'
    split = pd.read_csv(file_path)
    credits = pd.concat([credits, split], ignore_index=True)

movies_m = pd.read_csv('movies_m.csv')
print(movies.head())
print(ratings.head())
print(links.head())
print(keywords.head())


"""
Data Pre-processing and EDA
"""

"""
Clean the genres data and extract their names
"""

def get_genre_names(genre_list: List[dict]) -> List[str]:
    """
    Extracts the names of movie genres.
    """
    return [i['name'] for i in genre_list]

def clean_genre_column(df, column_name: str):
    """
    Cleans a column in a DataFrame by extracting the genre names for each movie.
    """
    df[column_name] = ( df[column_name].fillna('[]')   # Fill missing values with an empty list
        .apply(ast.literal_eval)       # Convert strings to Python lists
        .apply(get_genre_names) )      # Extract the genre names

# Call the function to clean the 'genres' column
clean_genre_column(movies, 'genres')
print(movies['genres'].head(10))


"""
Extract the release year
"""

def extract_year(date_str):
    """
    Extract the release year from the release date
    """
    if date_str != np.nan:
        return str(pd.to_datetime(date_str, errors='coerce').year)
    else:
        return np.nan

movies['year'] = movies['release_date'].apply(extract_year)




















"""
Top 50 movies recommendation in different popular genres
"""
# Build the overall Top 50 movies recommendation chart in different popular genres.

# Create a Weighted rating formula to get the result chart.
# Weighted Rating = [(v/v+m)*R] + [(m/v+m)*C]
# v is the number of votes for a movie. This is an indicator of how popular or well-known the movie is.
# m is the minimum number of votes required for a movie to be listed in the chart.
# R is the average rating of the movie.
# C is the mean vote across all movies in the dataset.

# Clean and convert the vote_count and vote_average column
vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')

# Calculate C
C = vote_averages.mean()

# Use 90% as the cutoff to calculate m
m = vote_counts.quantile(0.90)

# To get the qualified movies
Top_movies = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull())
            & (movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

# Convert vote_count to int and vote_average to float
Top_movies['vote_count'] = Top_movies['vote_count'].astype('int')
Top_movies['vote_average'] = Top_movies['vote_average'].astype('float')

# Create WR formula
def weighted_rating(x):
    """
    Calculates the weighted rating for a given movie based on its vote count,
    vote average, and the total mean vote and minimum vote count required.
    """
    # Extract the vote count and vote average for the movie
    v = x['vote_count']
    R = x['vote_average']

    # Calculate the weighted rating using the formula
    WR = (v / (v + m) * R) + (m / (m + v) * C)

    return WR

Top_movies['WR'] = Top_movies.apply(weighted_rating,axis = 1)
Top_movies = Top_movies.sort_values('WR', ascending= False).head(50)
print(Top_movies)


"""
Get the top 20 movies in some popular genres
"""

# We can use the WR formula to get the top 20 movies in different popular genres, like comedy, action and so on
# A movie may belong to different movie genres, so we need to split the genres column

def split_genres(movies):
    """
    split the genres into multiple rows
    and return a new one with one row per genre.
    """
    # Split the genres column into multiple rows using lambda function and stack() method
    s = movies.apply(lambda x: pd.Series(x['genres'], dtype='str'), axis=1).stack()

    # Drop the original index level and rename the series
    s = s.reset_index(level=1, drop=True).rename('genre')

    # Drop the genres column from the original dataframe and join with the new series
    movies_genre = movies.drop('genres', axis=1).join(s)

    return movies_genre

movies_genre = split_genres(movies)
# print(movies_genre)

# Create a function to get the top 20 movies in different genres with the new cutoff with 80%
def build_top(genre, percentile=0.8, genre_name = None):
    # Select all movies of the given genre from the preprocessed movies_genre
    genre_movies = movies_genre[movies_genre['genre'] == genre]

    # Calculate the mean vote average and the vote count threshold
    vote_averages = genre_movies[genre_movies['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_averages.mean()
    vote_counts = genre_movies[genre_movies['vote_count'].notnull()]['vote_count'].astype('int')
    m = vote_counts.quantile(percentile)

    # To get the qualified movies
    Top_20 = genre_movies[(genre_movies['vote_count'] >= m) & (genre_movies['vote_count'].notnull()) & (
        genre_movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]

    # Convert vote_count to int and vote_average to float
    Top_20['vote_count'] = Top_20['vote_count'].astype('int')
    Top_20['vote_average'] = Top_20['vote_average'].astype('float')

    # Calculate the weighted rating using the formula
    Top_20['WR'] = Top_20.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)

    Top_20 = Top_20.sort_values('WR', ascending=False).head(20)

    if genre_name:
        Top_20['genre'] = genre_name

    return Top_20


"""
Try to return the top 20 movies in some popular genres
"""

Action_20 = build_top('Action', genre_name='Action')
print(Action_20)

Drama_20 = build_top('Drama', genre_name='Drama')
print(Drama_20)

Adventure_20 = build_top('Adventure', genre_name='Adventure')
print(Adventure_20)



"""
Content Based Filtering 
"""

# Correlate the links to movies
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype('int')

movies_l = movies[movies['id'].isin(links_small)]
print(movies_l)

# 1. Using the descriptions and taglines of movies to recommend
movies_l['tagline'] = movies_l['tagline'].fillna('')
movies_l['description'] = movies_l['overview'] + movies_l['tagline']
movies_l['description'] = movies_l['description'].fillna('')

# Create a TfidfVectorizer object
tf = TfidfVectorizer(
    analyzer='word',     # Analyze at the word level
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=0,            # Include words that occur in at least 1 document
    stop_words='english' # Exclude English stop words
)

# Use the TfidfVectorizer to transform the 'description' column of the movies_l dataset
# into a matrix of tf-idf features
tfidf_matrix = tf.fit_transform(movies_l['description'])

print(tfidf_matrix.shape)
print(tfidf_matrix)


# Use dot product to get the Cosine Similarity.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Write a function to return the 15 most similar movies based on Cosine Similarity score
movies_l = movies_l.reset_index()
titles = movies_l['title']
indices = pd.Series(movies_l.index, index=movies_l['title'])


def content_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies = movies_l.iloc[movie_indices][['title', 'vote_average', 'genres', 'year', 'description']]

    return recommend_movies

print(content_recommendations('The Dark Knight Rises'))

print(content_recommendations('The Shawshank Redemption'))



# 2. Using the crew, cast and keywords to recommend

# Correlate the credits and keywords to movies
# Clean the id in keywords and credits
keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
keywords.dropna(subset=['id'], inplace=True)
keywords['id'] = keywords['id'].astype('int')

credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
credits.dropna(subset=['id'], inplace=True)
credits['id'] = credits['id'].astype('int')

movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

movies_k = movies[movies['id'].isin(links_small)]
print(movies_k.shape)



movies_k['cast'] = movies_k['cast'].apply(literal_eval)
movies_k['crew'] = movies_k['crew'].apply(literal_eval)
movies_k['keywords'] = movies_k['keywords'].apply(literal_eval)
movies_k['cast_size'] = movies_k['cast'].apply(lambda x: len(x))
movies_k['crew_size'] = movies_k['crew'].apply(lambda x: len(x))

# Get the director name
def get_director(crew_list):
    for crew_member in crew_list:

        if crew_member['job'] == 'Director':

            return crew_member['name']

    return np.nan

movies_k['director'] = movies_k['crew'].apply(get_director)


def get_actor_names(cast_list):
    if isinstance(cast_list, list):
        # Extract actor names from the list
        return [i['name'] for i in cast_list]
    else:
        # Return an empty list for non-list entries
        return []

movies_k['cast'] = movies_k['cast'].apply(get_actor_names)

def limit_actors(cast_list, max_actors=3):
    if len(cast_list) >= max_actors:
        # Return the top 3 actors
        return cast_list[:max_actors]
    else:
        # Return the entire list for less than 3 actors
        return cast_list

movies_k['cast'] = movies_k['cast'].apply(limit_actors)

def get_keywords(keywords_list):
    if isinstance(keywords_list, list):
        # Extract keyword names from the list
        return [i['name'] for i in keywords_list]
    else:
        # Return an empty list for non-list entries
        return []

movies_k['keywords'] = movies_k['keywords'].apply(get_keywords)


def clean_cast_names(cast_list):
    """
    Strip name spaces and convert to lowercase
    """
    cleaned_list = []
    for name in cast_list:
        cleaned_name = str.lower(name.replace(" ", ""))
        cleaned_list.append(cleaned_name)
    return cleaned_list

movies_k['cast'] = movies_k['cast'].apply(clean_cast_names)


def clean_director_name(name):
    """
    Strip  directors' name spaces and convert to lowercase
    """
    return str.lower(name.replace(" ", ""))

def add_director_weight(director_name):
    """
    Count director name 3 times to give it more weight relative to the cast
    """
    return [director_name, director_name, director_name,]

movies_k['director'] = movies_k['director'].astype('str').apply(clean_director_name)
movies_k['director'] = movies_k['director'].apply(add_director_weight)


def extract_keywords(dataset):
    """
    Extract keywords and reorganize them into a new dataset
    """
    keywords = dataset.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    keywords.name = 'keywords'
    return keywords

e_keywords = extract_keywords(movies_k)

# Remove the keywords which only occur once
e_keywords = e_keywords.value_counts()
e_keywords = e_keywords[e_keywords > 1]

# Using SnowballStemmer to reduce a word to its base or root form
def stem_keywords(keyword_list):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(keyword) for keyword in keyword_list]

# Create a function to filter keywords
def filter_keywords(keywords):
    words = []
    for i in keywords:
        if i in e_keywords:
            words.append(i)
    return words

# Strip keywords the spaces and convert to lowercase
def clean_keywords(keyword_list):
    return [str.lower(keyword.replace(" ", "")) for keyword in keyword_list]

movies_k['keywords'] = movies_k['keywords'].apply(filter_keywords)
movies_k['keywords'] = movies_k['keywords'].apply(stem_keywords)
movies_k['keywords'] = movies_k['keywords'].apply(clean_keywords)

movies_k['g_features'] = movies_k['genres'] + movies_k['director'] + movies_k['cast'] + movies_k['keywords']
movies_k['g_features'] = movies_k['g_features'].apply(lambda x: ' '.join(x))

# Create a new Count Vectorizer
cf = CountVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=0,
    stop_words='english'
)

count_matrix = cf.fit_transform(movies_k['g_features'])

# Get the Cosine Similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# Create a funtion to recommend based on general features
movies_k = movies_k.reset_index()
titles = movies_k['title']
indices = pd.Series(movies_k.index, index=movies_k['title'])


def content_recommendations_g(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies = movies_k.iloc[movie_indices][['title', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    # Only return once of the director's name
    recommend_movies['director'] = recommend_movies['director'].apply(lambda x: list(set(x))[0])

    return recommend_movies


print(content_recommendations_g('The Dark Knight Rises'))

print(content_recommendations_g('The Shawshank Redemption'))




# 3. Improve the content-based recommendation by adding indexs of pipolarity and ratings
def content_recommendations_improved(title,n):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:16]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies_improved = movies_k.iloc[movie_indices][
        ['title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    # Only return once of the director's name
    recommend_movies_improved['director'] = recommend_movies_improved['director'].apply(lambda x: list(set(x))[0])
    vote_counts = recommend_movies_improved[recommend_movies_improved['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = recommend_movies_improved[recommend_movies_improved['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_averages.mean()
    m = vote_counts.quantile(0)    # set the cutoff as 60%

    recommend = recommend_movies_improved[
        (recommend_movies_improved['vote_count'] >= m) & (recommend_movies_improved['vote_count'].notnull()) & (recommend_movies_improved['vote_average'].notnull())]
    recommend['vote_count'] = recommend['vote_count'].astype('int')
    recommend['vote_average'] = recommend['vote_average'].astype('float')
    recommend['WR'] = recommend.apply(weighted_rating, axis=1)
    recommend = recommend.sort_values('WR', ascending=False).head(n)
    return recommend


print(content_recommendations_improved('The Dark Knight Rises',n=10))

print(content_recommendations_improved('The Shawshank Redemption',n=10))





"""
Collaborative Filtering
"""




"""
User-Based Collaborative Filtering Recommendation
"""

movie_ratings = pd.merge(movies_m, ratings,on='movieId')

# Building User-Movie Rating Matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating')

# pivot table of user-movie ratings
movie_ratings_pivot = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

user_similarity_matrix = cosine_similarity(user_movie_matrix.fillna(0))

print(user_similarity_matrix.shape)
print(user_similarity_matrix)

# Defining a function to get K most similar users
def get_top_k_similar_users(target_user_id, k=5):
    target_user_similarity = user_similarity_matrix[target_user_id - 1]
    top_k_users = target_user_similarity.argsort()[::-1][1:k+1]
    top_k_similarity = target_user_similarity[top_k_users]
    return top_k_users, top_k_similarity

def get_recommendations_user(target_user_id, k=5, n=10):
    top_k_users, _ = get_top_k_similar_users(target_user_id, k=k)
    neighbors_movies = user_movie_matrix.loc[top_k_users]
    target_user_movies = user_movie_matrix.loc[target_user_id][user_movie_matrix.loc[target_user_id] > 0].index
    neighbors_movies = neighbors_movies.loc[:, ~neighbors_movies.columns.isin(target_user_movies)]
    neighbors_mean_rating = neighbors_movies.mean(axis=0)
    neighbors_mean_rating = neighbors_mean_rating.sort_values(ascending=False)
    recommendations = movies_m.loc[movies_m['movieId'].isin(neighbors_mean_rating.head(n).index)]

    return recommendations[['movieId', 'title', 'genres']]


# Test
target_user_id = 1
recommendations = get_recommendations_user(target_user_id, k=5, n=10)
print(recommendations)




"""
Item-Based Collaborative Filtering Recommendation
"""





# Build the rating matrix
ratings_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Calculate the similarity between movies
movie_similarity_matrix = 1 - pairwise_distances(ratings_matrix.T.values, metric='cosine')

print(movie_similarity_matrix.shape)
print(movie_similarity_matrix)

# Convert similarity matrix to DataFrame
movie_similarity = pd.DataFrame(movie_similarity_matrix, index=ratings_matrix.columns, columns=ratings_matrix.columns)


# Define function to get recommended movies
def get_recommendations_item(user_id, n):
    # Get unrated movies for the specified user
    user_ratings = ratings_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    # Calculate the recommendation score for each movie
    movie_scores = []
    for movie_id in unrated_movies:
        # Get the similarity between the movie and rated movies by the user
        similarity = movie_similarity[movie_id][user_ratings.index]
        # Calculate the recommendation score,
        # which is the sum of the product of similarity and rating divided by the sum of similarity
        score = np.sum(similarity * user_ratings) / np.sum(similarity)
        movie_scores.append((movie_id, score))
    # Sort movies by recommendation score and select the top n movies
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_movies = [movie[0] for movie in movie_scores[:n]]
    recommendations = movie_ratings.loc[movie_ratings['movieId'].isin(recommended_movies)]

    return recommendations[['movieId', 'title', 'genres']]

# Test
target_user_id = 1
recommendations = get_recommendations_item(target_user_id,n=10)
print(recommendations)







"""
Matrix Factorization
"""


"""
SVD (Singular Value Decomposition) Approach
"""

# read the SVD model
with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)


    
    
    
# link the movieId to id in the movies.csv
integrate_id = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]

# define a function to convert 'tmdbId' values to integers
def convert_id(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return np.nan

integrate_id['tmdbId'] = integrate_id['tmdbId'].apply(convert_id)
integrate_id.columns = ['movieId', 'id']

# merge the 'integrate_id' and 'movies_k' dataframes on the 'id' column and set the index to 'title'
integrate_id = integrate_id.merge(movies_k[['title', 'id']], on='id').set_index('title')

# create a dictionary that maps the 'id' column to its corresponding index in the 'cosine_sim' matrix
indices_integrate = integrate_id.set_index('id')

def hybrid(userId, title, n):
    idx = indices[title]
    # get the 'tmdbId' and 'movieId' of the input movie title
    tmdbId = integrate_id.loc[title]['id']
    movie_id = integrate_id.loc[title]['movieId']

    # compute the cosine similarities between the input movie and all other movies
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    # sort the similarities in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # select the top 30 most similar movies
    sim_scores = sim_scores[1:31]
    # get the indices of the most similar movies
    movie_indices = [i[0] for i in sim_scores]

    # select the relevant columns from the 'movies_k' dataframe for the most similar movies
    movies = movies_k.iloc[movie_indices][['id', 'title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    movies['movieId'] = movies['id']
    movies = movies.drop(columns=['id'])
    # predict the rating of the most similar movies for the input user using the SVD model
    movies['Predicted rating'] = movies['movieId'].apply(lambda x: svd.predict(userId, indices_integrate.loc[x]['movieId']).est)
    # sort the movies in descending order of predicted rating
    movies = movies[['movieId', 'title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords', 'Predicted rating']].sort_values('Predicted rating', ascending=False)
    
    return movies.head(n)

# Test
hybrid(98, 'The Dark Knight Rises',20)
