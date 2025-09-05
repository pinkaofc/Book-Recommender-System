import streamlit as st
import pickle
import numpy as np

# Custom CSS for improved pink theme, readability, and aesthetic background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&family=Comic+Neue:wght@700&display=swap');
    body, .stApp {
        background-image: url("https://i.pinimg.com/736x/57/6c/b4/576cb4d3a596e57c91d8d469707a30f8.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .main-title {
        font-family: 'Lobster', cursive;
        font-size: 3em;
        color: #ff1493;
        text-shadow: 2px 2px 8px #fff;
        text-align: center;
        margin-bottom: 0.5em;
        margin-top: 0.5em;
    }
    .section-header {
        font-family: 'Comic Neue', cursive;
        font-size: 1.5em;
        color: #d63384;
        text-align: center;
        margin-top: 2em;
        margin-bottom: 1em;
    }
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 2em;
    }
    .pink-btn button {
        background-color: #ff1493 !important;
        color: #fff !important;
        border-radius: 25px !important;
        border: none !important;
        font-family: 'Comic Neue', cursive !important;
        font-size: 18px !important;
        padding: 0.5em 2em !important;
        transition: background 0.3s;
        margin-top: 1em;
        margin-bottom: 2em;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .pink-btn button:hover {
        background-color: #d63384 !important;
        color: #fff !important;
    }
    .rec-card {
        background: #ffc1e3;
        color: #333333;
        border-radius: 18px;
        box-shadow: 0 4px 16px 0 #ffb6c1;
        padding: 1.2em 0.8em;
        margin: 0.8em 0;
        font-family: 'Comic Neue', cursive;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .rec-title {
        font-size: 1.15em;
        font-weight: bold;
        color: #c2185b;
        margin-bottom: 0.4em;
        font-family: 'Lobster', cursive;
        text-align: center;
    }
    .rec-author {
        font-size: 1em;
        color: #333333;
        font-style: italic;
        margin-bottom: 0.2em;
        text-align: center;
    }
    /* Center Streamlit selectbox */
    div[data-testid="stSelectbox"] > label, div[data-testid="stSelectbox"] > div {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    /* Center Streamlit button container */
    .pink-btn {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Book Recommender System 🎀</div>", unsafe_allow_html=True)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('book_pivot.pkl', 'rb') as f:
    book_pivot = pickle.load(f)

with open('books.pkl', 'rb') as f:
    books = pickle.load(f)


# Centered selectbox
selectbox_container = st.container()
with selectbox_container:
    book_list = list(book_pivot.index)
    selected_book = st.selectbox("", book_list, key="book_select")

# Custom styled button, centered
recommend_clicked = st.container()
with recommend_clicked:
    st.markdown('<div class="pink-btn">', unsafe_allow_html=True)
    recommend = st.button("Recommend 💖")
    st.markdown('</div>', unsafe_allow_html=True)

if recommend:
    indices = np.where(book_pivot.index == selected_book)[0]
    if len(indices) == 0:
        st.warning("Selected book not found in the dataset.")
    else:
        book_id = int(indices[0])
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
        )
        recommended_indices = suggestions[0][1:6]

       
        

        cols = st.columns(5)
        for idx, col in zip(recommended_indices, cols):
            rec_title = book_pivot.index[idx]
            meta = books[books['title'] == rec_title]
            if not meta.empty:
                meta = meta.iloc[0]
                author = meta.get('author', "Unknown Author")
            else:
                author = "Unknown Author"

            card_html = f"""
                <div class='rec-card'>
                    <div class='rec-title'>{rec_title}</div>
                    <div class='rec-author'>{author}</div>
                </div>
            """
            col.markdown(card_html, unsafe_allow_html=True)