import streamlit as st
import pandas as pd
import pickle
import movie_recommendation_system

st.set_page_config(page_title="Movie Recommendation System", layout="centered", page_icon=":film_projector:")
st.title('Movie Recommendation System :film_projector:')
st.sidebar.subheader('Movie Recommendation System :film_projector:')
st.sidebar.subheader('Jinbo Li')
option = st.sidebar.radio(
    "Find your movies",
    ("Get Recommendations", "EDA Parts", "TOP 50 MOVIES", "TOP 20 MOVIES in Popular Genres"))
st.sidebar.subheader(' :cinema: - About this Project')
st.sidebar.info("""     
-   With the increasing availability of movies on various platforms, the challenge of selecting the most preferred movies for users has become a critical issue in the entertainment industry.
-   The project aims to build a movie recommendation system to produce a ranked list of movies to provide personalized recommendations based on users’ preferences, rating history and other relevant factors. """)


if option == "Get Recommendations":
    st.info('Which recommender would you like to use?👇')

    left, mid, right = st.columns(3)

    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #00ff00; 
            color:#ff0000; 
        }
        div.stButton > button:hover {
            background-color: #0099ff;
            color:#ffffff;
            }
        </style>""", unsafe_allow_html=True)

    with left:
        with st.expander("Content-Based"):

            movie_title = st.text_input("Please input movie title")
            numbers_of_return = st.slider('Numbers of Recommendations', 5, 30, 5, 5, key=10)
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            get_recommendations = st.button('Get Recommendations-1 :tv:')

            def content(movie_title, n):
                try:
                    recommend = movie_recommendation_system.content_recommendations_improved(movie_title, n)
                    return recommend
                except KeyError:
                    return None

            if get_recommendations:
                if movie_title:
                    recommendations = content(movie_title, n=numbers_of_return)
                    if recommendations is not None:
                        st.write(pd.DataFrame(recommendations)[:numbers_of_return])
                    else:
                        st.warning('The movie you entered is not found in our database. '
                                   'Please make sure you have entered the correct movie title (including letter case).')
                else:
                    st.warning('Please enter a movie title.')

    with mid:
        with st.expander("Collaborative Filtering"):

            user_id = st.number_input("Please input user id", step=1)
            numbers_of_return = st.slider('Numbers of Recommendations', 5, 30, 5, 5, key=11)
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            get_recommendations2 = st.button('Get Recommendations-2 :tv:')

            def cf(user_id, k=5, n=numbers_of_return):
                recommend = movie_recommendation_system.get_recommendations_item(user_id, n=numbers_of_return)
                return recommend

            if get_recommendations2:
                if user_id:
                    recommendations2 = cf(user_id, k=5, n=numbers_of_return)
                    if recommendations2 is not None:
                        st.write(pd.DataFrame(recommendations2)[:numbers_of_return])
                    else:
                        st.warning('The user id you entered does not exist in our database.')
                else:
                    st.warning('Please enter a user id.')

    with right:
        with st.expander("Hybrid"):

            user_id2 = st.text_input("User id")
            movie_title2 = st.text_input("Movie title")
            numbers_of_return = st.slider('Numbers of Recommendations', 5, 30, 5, 5, key=12)
            get_recommendations3 = st.button('Get Recommendations-3 :tv:')

            def hy(user_id2, movie_title2, n=numbers_of_return):
                recommend = movie_recommendation_system.hybrid(user_id2, movie_title2, n=numbers_of_return)
                return recommend

            if get_recommendations3:
                if user_id2 and movie_title2:
                    recommendations3 = hy(user_id2, movie_title2, n=numbers_of_return)
                    if recommendations3 is not None:
                        st.write(pd.DataFrame(recommendations3)[:numbers_of_return])
                    else:
                        st.warning('The user id or movie you entered is not found in our database. '
                                   'Please make sure you have entered the correct information.')
                else:
                    st.warning('Please enter a user id and a movie title.')

if option == "EDA Parts":
    from PIL import Image
    image1 = Image.open('image2.png')
    st.image(image1)

    image2 = Image.open('image1.png')
    st.image(image2)

    image3 = Image.open('image3.png')
    st.image(image3)

    image4 = Image.open('image4.png')
    st.image(image4)

    image5 = Image.open('image5.png')
    st.image(image5)

    image6 = Image.open('image6.png')
    st.image(image6)

    image7 = Image.open('image7.png')
    st.image(image7)

    image8 = Image.open('image8.png')
    st.image(image8)

if option == "TOP 50 MOVIES":
    st.subheader('TOP 50 MOVIES')
    TOP_50 = movie_recommendation_system.Top_movies
    st.write(pd.DataFrame(TOP_50))

if option == "TOP 20 MOVIES in Popular Genres":
    m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #00ff00; 
                color:#ff0000; 
            }
            div.stButton > button:hover {
                background-color: #0099ff;
                color:#ffffff;
                }
            </style>""", unsafe_allow_html=True)

    Genres = st.selectbox(
        'Genres',
        ('Please select a genre', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy',
         'Family', 'History', 'Horror', 'Mystery', 'Romance', 'Science Fiction',
         'Thriller', 'War'))

    st.write("\n")
    st.write("\n")
    st.write("\n")
    get_top_20 = st.button('Get Top 20 movies :tv:')

    def top20(genres, percentile=0.8, genre_name=Genres):
        top = movie_recommendation_system.build_top(genres, percentile=0.8, genre_name=Genres)
        return top

    if get_top_20:
        if Genres:
            top_20 = top20(Genres, percentile=0.8, genre_name=Genres)
            st.write(pd.DataFrame(top_20))
        else:
            st.warning('Please select a genre.')
