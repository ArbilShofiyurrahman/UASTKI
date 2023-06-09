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
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Sistem Pencarian Menggunakan Metode BM25 Dalam Dokumen CISI",
        page_icon="🔎",
    )

    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    corpus = load_docs()

    bm25_simple, bm25_okapi, bm25_plus = load_models()

    st.header(':mag_right: Sistem Pencarian Menggunakan Metode BM25 Dalam Dokumen CISI')

    st.markdown('''
        <a href="https://github.com/tcvieira/bm25-exercise-report" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30" alt="github repository"></img>
        </a>git repository
        ''', unsafe_allow_html=True)

    st.markdown('---')
    
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
            doc_text += l.strip()[3:] + " "
    return list(doc_set.values())


@st.cache_resource
def load_models():

    bm25_okapi_file, _ = urllib.request.urlretrieve(
        'https://github.com/ArbilShofiyurrahman/UAS/blob/main/bm25-exercise-report-main/models/BM25Okapi.pkl?raw=true', 'bm25_okapi_file.downloaded')
    with open(bm25_okapi_file, 'rb') as file:
        bm25_okapi: BM25Okapi = pickle.load(file)
        print(bm25_okapi.corpus_size)

    bm25_plus_file, _ = urllib.request.urlretrieve(
        'https://github.com/ArbilShofiyurrahman/UAS/blob/main/bm25-exercise-report-main/models/BM25Plus.pkl?raw=true', 'bm25_plus_file.downloaded')
    with open(bm25_plus_file, 'rb') as file:
        bm25_plus: BM25Plus = pickle.load(file)
        print(bm25_plus.corpus_size)

    print(subprocess.run(['ls -la'], shell=True))
    st.success("BM25 models loaded!", icon='✅')
    return bm25_simple, bm25_okapi, bm25_plus


if __name__ == "__main__":
    main()
