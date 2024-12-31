import streamlit as st
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', "")
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else ""

# Function to fetch movie rating
def fetch_rating(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    data = requests.get(url).json()
    return data.get("vote_average", "N/A"), data.get("vote_count", 0)

# Function to fetch movie trailer
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    data = requests.get(url).json()
    for video in data['results']:
        if video['site'] == 'YouTube' and video['type'] == 'Trailer':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return ""

# Load movie data
movies = pickle.load(open("movies_list.pkl", 'rb'))
movies_list = movies['title'].values

# Generate similarity matrix dynamically using TF-IDF vectorizer
def generate_similarity_matrix():
    # Ensure all 'tags' are strings (handle NaNs or other non-string values)
    movies['tags'] = movies['tags'].fillna('').astype(str)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Page layout
st.set_page_config(page_title="CineVerse", layout="wide")

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }
    .header {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        color: #333;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .movie-title {
        font-size: 14px;
        font-weight: bold;
        color: #444;
        text-align: center;
        margin-top: 10px;
    }
    .footer {
        margin-top: 40px;
        font-size: 12px;
        color: #888;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='header'>🎬 Movie Recommender System -- CineVerse 🍿</div>", unsafe_allow_html=True)

# Select box for movie input
st.markdown("<div class='subheader'>Select a movie to get recommendations:</div>", unsafe_allow_html=True)
selectvalue = st.selectbox("Choose a movie", movies_list)

# Recommendation function using dynamic similarity matrix
def recommend(movie, filtered_movies):
    similarity_matrix = generate_similarity_matrix()  # Generate the similarity matrix on-the-fly
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity_matrix[index])),
        reverse=True,
        key=lambda vector: vector[1]
    )
    
    recommended_movies = []
    recommended_posters = []
    for i in distances[1:6]:  # Top 5 recommendations
        movie_id = movies.iloc[i[0]]['id']
        if movies.iloc[i[0]]['title'] in filtered_movies['title'].values:
            recommended_movies.append(movies.iloc[i[0]]['title'])
            recommended_posters.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_posters

# Carousel of trending movies
st.markdown("<div class='subheader'>🔥 Trending Movies:</div>", unsafe_allow_html=True)
image_urls = [
    fetch_poster(1632),
    fetch_poster(299536),
    fetch_poster(17455),
    fetch_poster(2830),
    fetch_poster(429422),
]

# Trending movies layout
cols = st.columns(len(image_urls))
for col, image_url in zip(cols, image_urls):
    with col:
        st.image(image_url, use_container_width=True)

# Personalized Recommendations
st.markdown("<div class='subheader'>Customize Your Recommendations:</div>", unsafe_allow_html=True)
genres = st.multiselect("Pick your favorite genres:", ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"])

# Filter recommendations based on genres
filtered_movies = movies[movies['tags'].str.contains("|".join(genres), na=False)]

# Show recommendations on button click
if st.button("Show Recommendations"):
    movie_names, movie_posters = recommend(selectvalue, filtered_movies)

    # Display recommendations with additional details
    st.markdown("<div class='subheader'>Recommended Movies 🎥</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for col, movie_name, movie_poster in zip(cols, movie_names, movie_posters):
        with col:
            st.image(movie_poster, use_container_width=True)
            st.markdown(f"<p class='movie-title'>{movie_name}</p>", unsafe_allow_html=True)

    # Display ratings and trailers for recommended movies
    for movie_name, movie_poster in zip(movie_names, movie_posters):
        movie_id = movies[movies['title'] == movie_name]['id'].values[0]
        rating, votes = fetch_rating(movie_id)
        trailer_url = fetch_trailer(movie_id)
        st.write(f"{movie_name}: ⭐ {rating} ({votes} votes)")
        if trailer_url:
            st.markdown(f"[Watch Trailer]({trailer_url})", unsafe_allow_html=True)

# Feedback loop (like/dislike recommendations)
if 'liked_movies' not in st.session_state:
    st.session_state.liked_movies = []

if st.button("👍 Like"):
    st.session_state.liked_movies.append(selectvalue)
st.write("Movies you liked:", st.session_state.liked_movies)

# Watchlist feature
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if st.button("Add to Watchlist"):
    st.session_state.watchlist.append(selectvalue)
st.write("Your Watchlist:", st.session_state.watchlist)

# Footer
st.markdown(
    """
    <hr>
    <div class='footer'>
        Created by Mohsin Irfan | © 2024 CineVerse
    </div>
    """,
    unsafe_allow_html=True
)
