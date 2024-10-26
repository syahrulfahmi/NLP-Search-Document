import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk membaca semua file dalam folder


def load_documents(folder_path):
    documents = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding="utf-8") as file:
                documents.append(file.read())
                file_names.append(filename)
    return documents, file_names

# Fungsi untuk menghitung TF-IDF dan similarity


def search_documents(query, documents, file_names):
    # Menggabungkan dokumen dan query untuk menghitung TF-IDF
    combined_docs = documents + [query]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_docs)

    # Hitung cosine similarity antara query dan dokumen
    cosine_similarities = cosine_similarity(
        tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Urutkan dokumen berdasarkan similarity tertinggi
    related_docs_indices = cosine_similarities.argsort()[::-1]
    results = [(file_names[i], cosine_similarities[i])
               for i in related_docs_indices if cosine_similarities[i] > 0]
    return results


# Streamlit App
st.title("Sistem Pencarian Dokumen Menggunakan TF-IDF")

# Path folder dokumen
folder_path = "docs"
documents, file_names = load_documents(folder_path)

# Input query pengguna
query = st.text_input("Masukan kata pencarian:")

# Jika query diinputkan, lakukan pencarian
if query:
    results = search_documents(query, documents, file_names)
    if results:
        data_dict = {
            "No": list(range(1, len(results) + 1)),
            "Nama Dokumen": [item[0] for item in results],
            "Score Cosinus Similarity": [float(item[1]) for item in results]
        }
        df = pd.DataFrame(data_dict)
        st.title("Tabel Dokumen dengan Nilai Cosinus Similarity")
        st.table(data_dict)
    else:
        st.write("Hasil pencarian tidak ditemukan")
