import pandas as pd
import numpy as np
import os, re, spacy
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from CIRS.processor import eliminate_stopwords

#
NUM_PLOTS_FOUND = 10

# Flag to control if the TF-IDF computation will be done using external libraries
use_external_libraries = False

# Definition of critical directories
SKRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.dirname(SKRIPT_DIR)
DATA_DIR = SKRIPT_DIR+"/IndexFiles.index"
FILE1_EXT_LIB = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
FILE2_EXT_LIB = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
FILE1_CUSTOM = os.path.join(DATA_DIR, "df_idf.csv")
FILE2_CUSTOM = os.path.join(DATA_DIR, "df_tf_idf.csv")

# Load preprocessed data and vectorizer
with open(FILE1_EXT_LIB, 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open(FILE2_EXT_LIB, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

#
df_movies = pd.read_csv(IN_DIR+"/Movie_Plots.csv")

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')


# Preprocessing function
def preprocess_text_english(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    stops_nltk = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stops_nltk]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


# Search function
def search_plots(qr, tfidf_vector, tfidf_mat):
    query_processed = preprocess_text_english(qr)
    query_vector = tfidf_vector.transform([query_processed])
    similarities = cosine_similarity(query_vector, tfidf_mat).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_tit = df_movies.iloc[sorted_indices]['Title']
    sorted_plt = df_movies.iloc[sorted_indices]['Plot']
    return sorted_tit, sorted_plt, similarities[sorted_indices]


#
if __name__ == '__main__':
    # Get user input for the search query
    query = input("Enter your search query: ")

    if use_external_libraries:
        if not os.path.isfile(FILE1_EXT_LIB) or not os.path.isfile(FILE2_EXT_LIB):
            print("Necessary files not found. Make sure to run the indexing first!")
        else:
            # Perform search
            sorted_titles, sorted_plots, sorted_similarities = search_plots(query, tfidf_vectorizer, tfidf_matrix)
            # Display results
            print("Search results for the query:", query)
            title = ""
            plot = ""
            similarity = 0.
            for title, plot, similarity in zip(sorted_titles.head(NUM_PLOTS_FOUND), sorted_plots.head(NUM_PLOTS_FOUND), sorted_similarities[:NUM_PLOTS_FOUND]):
                print(f"Title: {title}, Similarity: {similarity:.4f}")
                print(f"Plot: {plot}")
                print("------")
    else:
        if not os.path.isfile(FILE1_CUSTOM) or not os.path.isfile(FILE2_CUSTOM):
            print("Necessary files not found. Make sure to run the indexing first!")
        else:
            #
            df_idf = pd.read_csv(FILE1_CUSTOM, index_col=0)
            df_tf_idf = pd.read_csv(FILE2_CUSTOM, index_col=0)
            #
            cleansed_query = [query]
            eliminate_stopwords(cleansed_query)
            #
            # feature_idf = df_idf[]
            # print(df_tf_idf)


