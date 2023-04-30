import streamlit as st
import pandas as pd
import pickle




st.set_page_config(page_title="Movie Recommendation System",layout="centered",page_icon=":film_projector:")
st.title('Movie Recommendation System :film_projector:')
st.sidebar.subheader('Movie Recommendation System :film_projector:')
st.sidebar.subheader('Jinbo Li')
option=st.sidebar.radio(
    "Find your movies",
    ("Get Recommendations","EDA Parts", "TOP 50 MOVIES", "TOP 20 MOVIES in Popular Genres"))
st.sidebar.subheader(' :cinema: - About this Project')
st.sidebar.info( """     
-   With the increasing availability of movies on various platforms, the challenge of selecting the most preferred movies for users has become a critical issue in the entertainment industry.
-   The project aims to build a movie recommendation system to produce a ranked list of movies to provide personalized recommendations based on users’ preferences, rating history and other relevant factors. """)



if option == "Get Recommendations":


    st.info('Which recommender would you like to use?👇')



    left,mid, right=st.columns(3)

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
        st.success("Content-Based")

        movie_title = st.text_input("Please input movie title")
        numbers_of_return = st.slider('Numbers of Recommendations',10,30,15,5, key = 10)
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        get_recommendations = st.button('Get Recommendations-1 :tv:')



        with open("content_recommendations_improved.pkl", 'rb') as f:
            content = pickle.load(f)

        if get_recommendations:
            if movie_title:
                recommendations = content(movie_title)[:numbers_of_return]
                st.write(pd.DataFrame(recommendations))

            else:
                st.warning('Please enter a movie title.')




    with mid:
        st.success("Collaborative Filtering")


        user_id = st.text_input("Please input user id")
        numbers_of_return = st.slider('Numbers of Recommendations', 10, 30, 15, 5, key=11)
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        get_recommendations2 = st.button('Get Recommendations-2 :tv:')

    with right:
        st.success("Hybrid")

        user_id2 = st.text_input("User id")
        movie_title2 = st.text_input("Movie title")
        numbers_of_return = st.slider('Numbers of Recommendations', 10, 30, 15, 5, key=12)
        get_recommendations3 = st.button('Get Recommendations-3 :tv:')


