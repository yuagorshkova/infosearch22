import streamlit as st
from setup_instruments import bm25
from setup_instruments import tfidf
from setup_instruments import bert_search
from style import add_bg_from_local, header, text_w_background

if __name__ == "__main__":

    add_bg_from_local("background.jpg")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    header("Это поисковик!!", size=60)
    query = st.text_input("Введите запрос:")
    n = st.slider("Сколько вывести результатов?", 1, 50, 5)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        b1 = st.button("Искать с BM25")
    with col2:
        b2 = st.button("Искать с TF-iDF")
    with col3:
        b3 = st.button("Искать с BERT")

    if b1:
        selected_method = "BM25"
        search_result, total_time = bm25.time_find_closest_docs(query, n)
        header(selected_method, size=48, color="621B32")
        st.write(total_time)
        for item in search_result:
            text_w_background(item)
    if b2:
        selected_method = "TF-iDF"
        search_result, total_time = tfidf.time_find_closest_docs(query, n)
        header(selected_method, size=48, color="621B32")
        st.write(total_time)
        for item in search_result:
            text_w_background(item)
    if b3:
        selected_method = "BERT"
        search_result, total_time = bert_search.time_find_closest_docs(query, n)
        header(selected_method, size=48, color="621B32")
        st.write(total_time)
        for item in search_result:
            text_w_background(item)
