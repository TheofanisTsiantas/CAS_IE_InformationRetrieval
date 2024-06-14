import csv
import math
import textwrap

import pandas as pd
import numpy as np
import os, re, spacy
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from CIRS.processor import eliminate_stopwords

#
N_PLOTS = 10

# Flag to control if the TF-IDF computation will be done using external libraries
use_external_libraries = False

# Definition of critical directories
SKRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.dirname(SKRIPT_DIR)
DATA_DIR = SKRIPT_DIR+"/IndexFiles.index"
FILE1_EXT_LIB = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
FILE2_EXT_LIB = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
FILE_IDF = os.path.join(DATA_DIR, "idf.csv")
FILE_TF_IDF = os.path.join(DATA_DIR, "tf_idf.csv")
FILE_ALPHAS = os.path.join(DATA_DIR, "alphas.csv")

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

    if use_external_libraries:
        print("Information retrieval using external libraries!")
        if not os.path.isfile(FILE1_EXT_LIB) or not os.path.isfile(FILE2_EXT_LIB):
            print("Necessary files not found. Make sure to run the indexing first!")
        else:
            while True:
                # Get user input for the search query
                query = input("Enter your search query or hit 'e' to exit: ")
                if query == 'e':
                    exit(0)
                # Perform search
                sorted_titles, sorted_plots, sorted_similarities = search_plots(query, tfidf_vectorizer, tfidf_matrix)
                # Display results
                print("Search results for the query:", query)
                title = ""
                plot = ""
                similarity = 0.
                for title, plot, similarity in zip(sorted_titles.head(N_PLOTS), sorted_plots.head(N_PLOTS), sorted_similarities[:N_PLOTS]):
                    print(f"Title: {title}, Similarity: {similarity:.4f}")
                    print(f"Plot: {plot}")
                    print("------")
    else:
        print("Information retrieval using custom code!")
        if not os.path.isfile(FILE_IDF) or not os.path.isfile(FILE_TF_IDF) or not os.path.isfile(FILE_ALPHAS):
            print("Necessary files not found. Make sure to run the indexing first!")
        else:
            # Read the IDF csv file to a dictionary
            print("Reading IDF data...")
            with open(FILE_IDF, mode='r') as file:
                csv_reader = csv.reader(file)
                idf_dic = {}
                for data in csv_reader:
                    idf_dic[data[0]] = data[1]
            # Read the IDF csv file to a dictionary
            print("Reading TF-IDF data...")
            with open(FILE_TF_IDF, mode='r') as file:
                csv_reader = csv.reader(file)
                tf_idf_dic = {}
                for data in csv_reader:
                    tf_idf_dic[data[0]] = {data[1]: data[2]}
                    for data_idx in range(3, len(data), 2):
                        tf_idf_dic[data[0]][data[data_idx]] = [data[data_idx + 1]]
            # Read the alphas csv file to a dictionary
            print("Reading alphas data...")
            with open(FILE_ALPHAS, mode='r') as file:
                csv_reader = csv.reader(file)
                alphas = {}
                for data in csv_reader:
                    alphas[data[0]] = data[1]

            while True:
                # Get user input for the search query
                query = input("Enter your search query or hit 'e' to exit: ")
                if query == 'e':
                    exit(0)
                cleansed_query = [query]
                eliminate_stopwords(cleansed_query)
                #
                cleansed_query = cleansed_query[0].split()
                features = set()
                for word in cleansed_query:
                    if word in idf_dic:
                        features.add(word)

                #
                relevant_documents = set()
                for feature in features:
                    for doc in tf_idf_dic[feature]:
                        relevant_documents.add(doc)

                RSV_dic = {}
                for doc in relevant_documents:
                    alpha_beta = 0
                    norm_b = 0
                    for feature in features:
                        beta = cleansed_query.count(feature)*float(idf_dic[feature])
                        if doc in tf_idf_dic[feature]:
                            alpha = float(tf_idf_dic[feature][doc][0])
                        else:
                            alpha = 0.
                        alpha_beta += beta*alpha
                        norm_b += math.pow(beta,  2)
                    alpha = float(alphas[doc])
                    RSV_dic[doc] = alpha_beta/(math.sqrt(norm_b) * alpha)

                #
                RSV_dic = dict(sorted(RSV_dic.items(), key=lambda item: item[1], reverse=True))
                RSV_dic = dict(list(RSV_dic.items())[:N_PLOTS])

                for plot_idx, score in RSV_dic.items():
                    idx = int(plot_idx)
                    title = df_movies['Title'].loc[idx]
                    year = df_movies['Release Year'].loc[idx]
                    plot = df_movies['Plot'].loc[idx]
                    print()
                    print(f"Title: {title}, Year: {year}, Score: {score}")
                    print()
                    print("PLOT:")
                    print()
                    print(textwrap.fill(plot, width=50))
