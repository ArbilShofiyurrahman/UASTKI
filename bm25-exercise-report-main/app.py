import subprocess
import urllib
import os
import pickle
import time
import streamlit as st
from rank_bm25 import BM25Okapi, BM25Plus
from bm25Simple import BM25Simple

path = os.path.dirname(__file__)
print(path)
print(subprocess.run(['ls -la'], shell=True))
print()
print(subprocess.run(['ls -la models/'], shell=True))
print()
print(subprocess.run(['ls -la content/'], shell=True))


def main():

    st.set_page_config(
        # Can be "centered" or "wide". In the future also "dashboard", etc.
        layout="wide",
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        # String or None. Strings get appended with "• Streamlit".
        page_title="Sistem Pencarian Menggunakan Metode BM25 Dalam Dokumen CISI",
        page_icon="🔎",  # String, anything supported by st.image, or None.
    )

    # LAYOUT
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    # padding = 2
    # st.markdown(f""" <style>
    #     .reportview-container .main .block-container{{
    #         padding-top: {padding}rem;
    #         padding-right: {padding}rem;
    #         padding-left: {padding}rem;
    #         padding-bottom: {padding}rem;
    #     }} </style> """, unsafe_allow_html=True)

    # horizontal radios
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # load documents
    corpus = load_docs()

    # load models
    bm25_simple, bm25_okapi, bm25_plus = load_models()

    # UI
    # st.header(f':mag_right: {algo}')
    st.header(':mag_right: BM25 based Information Retrieval System')

    st.markdown('''
        <a href="https://github.com/tcvieira/bm25-exercise-report" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30" alt="github repository"></img>
        </a>git repository
        ''', unsafe_allow_html=True)

    st.markdown('---')

    # Sidebar
    st.sidebar.markdown('---')
    st.sidebar.markdown('# About CISI')

    st.sidebar.markdown('''
        The CISI (Computer and Information Science Index) is a collection of documents used in information retrieval research. This system utilizes the BM25 method for information retrieval and searches through the CISI document corpus. You can enter a query in the search bar and see the most relevant documents retrieved by different BM25 variants.

        **Query Examples**
        - What system combines multiprogramming or remote station in information retrieval? What will be the extent of its use in the future?
        - What problems and concerns exist in making descriptive titles? What difficulties are involved in automatically taking articles from approximate titles?
        - What is information science? Provide a definition if possible.
        - Some considerations regarding the cost-effectiveness of online services in libraries
        - Fast Procedures for Computing Similarity Coefficients on Automatic Classification

        ''')

    with st.form("search_form"):
        query = st.text_input(
            'Query', 'IPING GILA IPING GILA IPING GILA')
        st.caption('no text preprocessing')

        submitted = st.form_submit_button('Search')

    if submitted:
        if query:
            st.markdown('---')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('BM25 Simple')

                bm25_simple_time, most_relevant_documents = search_docs(
                    bm25_simple, query, corpus)
                st.caption(f'time: {bm25_simple_time}')
                print_docs(most_relevant_documents)

            with col2:
                st.subheader('BM25OKapi')

                bm25_okapi_time, most_relevant_documents = search_docs(
                    bm25_okapi, query, corpus)
                st.caption(f'time: {bm25_okapi_time}')
                print_docs(most_relevant_documents)

        else:
            st.text('add some query')


def search_docs(model, query, corpus):
    tokenized_query = query.split(" ")

    start = time.time()
    most_relevant_documents = model.get_top_n(
        tokenized_query, corpus, 20)
    elapsed = (time.time() - start)
    return elapsed, most_relevant_documents[:20]


def print_docs(docs):
    for index, doc in enumerate(docs):
        st.markdown(f'''
                    <div style="text-align: justify">
                    <strong>{index+1}</strong>: {doc}
                    </div>
                    </br>
                    ''', unsafe_allow_html=True)


@st.cache_resource
def load_docs():
    # Processing DOCUMENTS
    doc_set = {}
    doc_id = ""
    doc_text = ""
    documents_file, _ = urllib.request.urlretrieve(
        'https://gist.githubusercontent.com/ArbilShofiyurrahman/d1a30628edd10df04169478f52b512fd/raw/6ea5546d3455b1376c7be6f448908ab46dee41eb/CISI.ALL', 'CISI.ALL.downloaded')
    with open(documents_file) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    for l in lines:
        if l.startswith(".I"):
            doc_id = int(l.split(" ")[1].strip())-1
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            # The first 3 characters of a line can be ignored.
            doc_text += l.strip()[3:] + " "
    return list(doc_set.values())


@st.cache_resource
def load_models():

    bm25_simple_file, _ = urllib.request.urlretrieve(
        'https://github.com/tcvieira/bm25-exercise-report/blob/main/models/BM25_simple.pkl?raw=true', 'bm25_simple_file.downloaded')
    with open(bm25_simple_file, 'rb') as file:
        bm25_simple: BM25Simple = pickle.load(file)
        print(bm25_simple.corpus_size)

    bm25_okapi_file, _ = urllib.request.urlretrieve(
        'https://github.com/ArbilShofiyurrahman/UAS/blob/main/bm25-exercise-report-main/models/BM25Okapi.pkl?raw=true', 'bm25_okapi_file.downloaded')
    with open(bm25_okapi_file, 'rb') as file:
        bm25_okapi: BM25Okapi = pickle.load(file)
        print(bm25_okapi.corpus_size)

    return bm25_simple, bm25_okapi


if __name__ == "__main__":
    main()
