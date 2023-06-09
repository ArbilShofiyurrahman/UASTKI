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
    bm25_okapi, bm25_plus = load_models()

    # UI
    # st.header(f':mag_right: {algo}')
    st.header(':mag_right: Sistem Pencarian Menggunakan Metode BM25 Dalam Dokumen CISI')

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
        The CISI CISI adalah singkatan dari "Commonwealth Institute of Science and Industry". CISI adalah sebuah institusi yang didirikan di Britania Raya pada tahun 1964 dengan tujuan mengembangkan ilmu pengetahuan dan teknologi serta mendorong inovasi dalam berbagai bidang seperti industri, sains, dan teknologi. Institusi ini berfokus pada riset, pendidikan, dan kolaborasi dengan industri untuk memajukan pengetahuan dan menghasilkan dampak positif bagi masyarakat.

    with st.form("search_form"):
        query = st.text_input(
            'Query', 'IPING GILA IPING GILA IPING GILA')
        st.caption('no text preprocessing')

        with st.expander("Query Examples"):
            st.markdown('''
                        - Sistem apa yang menggabungkan multiprogramming atau stasiun jarak jauh dalam pencarian informasi? Apa yang akan menjadi sejauh mana penggunaannya di masa depan?
                        - Masalah dan kekhawatiran apa yang ada dalam membuat judul deskriptif? Kesulitan apa yang terlibat dalam mengambil artikel secara otomatis dari perkiraan judul?
                        - Apa itu ilmu informasi? Berikan definisi jika memungkinkan.
                        - Beberapa Pertimbangan Terkait Keefektifan Biaya Layanan Online di Perpustakaan
                        - Prosedur Cepat Perhitungan Koefisien Kesamaan pada Klasifikasi Otomatis
                        ''')

        submitted = st.form_submit_button('Search')

    if submitted:
        if query:
            st.markdown('---')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('BM25OKapi')

                bm25_okapi_time, most_relevant_documents = search_docs(
                    bm25_okapi, query, corpus)
                st.caption(f'time: {bm25_okapi_time}')
                print_docs(most_relevant_documents)

            with col2:
                st.subheader('BM25+')

                bm25_plus_time, most_relevant_documents = search_docs(
                    bm25_plus, query, corpus)
                st.caption(f'time: {bm25_plus_time}')
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
    # OKAPI BM25
    model_file, _ = urllib.request.urlretrieve(
        'https://gist.githubusercontent.com/ArbilShofiyurrahman/ea2f1d67fa0d6debc10f800d73eb132a/raw/024c8eb8d1260f649d4c0e56b7b96ce8a17e2d61/cisi_bm25Okapi', 'cisi_bm25Okapi.downloaded')
    with open(model_file, 'rb') as f:
        bm25_okapi = pickle.load(f)

    # BM25+
    model_file, _ = urllib.request.urlretrieve(
        'https://gist.githubusercontent.com/ArbilShofiyurrahman/6be141fcf7bb4fd7d57b1546503b3ae6/raw/4b92a8c47b583ed8906484f2d4dbd09d133d7b5e/cisi_bm25Plus', 'cisi_bm25Plus.downloaded')
    with open(model_file, 'rb') as f:
        bm25_plus = pickle.load(f)

    return bm25_okapi, bm25_plus


if __name__ == '__main__':
    main()
