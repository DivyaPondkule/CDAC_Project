import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="centered",
    # initial_sidebar_state="expanded",
)

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Top 50 Books", "Books Popular yearly", "Analysis"],
                           icons=['house', "list-task", "list-task", "gear"],
                           menu_icon="cast", default_index=0)
    selected
    # selected = option_menu(None, ["Home", "Top 50 Books", "Books Popular yearly", "Analysis"],
    #                        icons=['house', "list-task", "list-task", "gear"],
    #                        menu_icon="cast", default_index=0, orientation="horizontal")
    # selected

# selected = option_menu(None, ["Home", "Top 50 Books", "Books Popular yearly", "Analysis"],
#     icons=['house', "list-task", "list-task", "gear"],
#     menu_icon="cast", default_index=0, orientation="horizontal")
# selected

if selected== "Home":
    st.header("Book Recommender system")

    model_knn = pickle.load(open('model_knn.pkl', 'rb'))
    books_name = pickle.load(open('books_name.pkl', 'rb'))
    user_rating = pickle.load(open('user_rating.pkl', 'rb'))
    popular_df = pickle.load(open('popular_df.pkl', 'rb'))
    user_rating_pivot2 = pickle.load(open('user_rating_pivot2.pkl', 'rb'))

    selected_books = st.selectbox(
        "Type or select a book",
        books_name
    )


    def fetch_poster(suggestion):
        book_name = []
        ids_index = []
        poster_url = []

        for book_id in suggestion:
            book_name.append(user_rating_pivot2.index[book_id])

        for name in book_name[0]:
            ids = np.where(user_rating['Book-Title'] == name)[0][0]
            ids_index.append(ids)

        for ids in ids_index:
            url = user_rating.iloc[ids]['Image-URL-M']
            poster_url.append(url)
        return poster_url


    def recommend_books(bk_name):
        book_list = []
        # index fetch
        if bk_name == ' ':
            st.text('Select a book')
        else:
            book_id = np.where(user_rating_pivot2.index == bk_name)[0][0]
            distance, suggestion = model_knn.kneighbors(user_rating_pivot2.iloc[book_id, :].values.reshape(1, -1),
                                                        n_neighbors=6)
            poster_url = fetch_poster(suggestion)
            for i in range(len(suggestion)):
                books = user_rating_pivot2.index[suggestion[i]]
                for j in books:
                    book_list.append(j)
                return book_list, poster_url


    if st.button('Show Recommendations'):
        recommendation_books, poster_url = recommend_books(selected_books)
        for i in range(1, 6):
            st.text(str(i) + ". " + recommendation_books[i])
            st.image(poster_url[i])

if selected == "Top 50 Books":
    st.header("Top 50 books")
    popular_df = pickle.load(open('artifacts/popular_df.pkl', 'rb'))

    for i in range(len(popular_df)):
        st.text(str(i + 1) + ". " + popular_df.iloc[i][1])
        st.image(popular_df.iloc[i][3])

if selected ==  "Books Popular yearly":

    st.header("Books popular yearly")
    popular_df_y = pickle.load(open('artifacts/popular_df_y.pkl', 'rb'))

    for i in range(len(popular_df_y)):
        # st.text(str(i+1)+". "+popular_df_y.iloc[i][1]+" Year: "+str(popular_df_y.iloc[i][3]))
        st.text(str(i + 1) + ". " + " Year: " + str(popular_df_y.iloc[i][3]))
        st.text(popular_df_y.iloc[i][1])

if selected == "Analysis":
    st.header("Analysis")

    image1 = Image.open('pages/A_1.png')
    st.image(image1, caption='Analysis 1 - Author with highest no.of books published')

    image2 = Image.open('pages/A_2.png')
    st.image(image2, caption='Analysis 2 - Top publishers')

    image3 = Image.open('pages/A_3.png')
    st.image(image3, caption='Analysis 3 - Top 10 highest rated books')

    image4 = Image.open('pages/A_4.png')
    st.image(image4, caption='Analysis 4 - Top 10 highest rated authors')

    image5 = Image.open('pages/A_5.png')
    st.image(image5, caption='Analysis 5 - Number of Books published on yearly basis')

    image6 = Image.open('pages/A_6.png')
    st.image(image6, caption='Analysis 6 - Age distribution')

    # st.markdown("<h3 style='text-align: center; color: white;'>Smaller headline in black </h2>", unsafe_allow_html=True)
    image7 = Image.open('pages/A_7.png')
    st.image(image7, caption='Analysis 7 - Rating distribution')
