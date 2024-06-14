# Necessary packages
import csv
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re
from CIRS.processor import eliminate_stopwords
import numpy as np


# Flag to control if the TF-IDF computation will be done using external libraries
use_external_libraries = False

# Definition of critical directories
SKRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.dirname(SKRIPT_DIR)
OUT_DIR = SKRIPT_DIR+"/IndexFiles.index"


# TF-IDF computation using external libraries
def create_tf_idf_ext_lib(plots: list):
    print("Starting vectorization with external libraries...")
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    df_tfidf = tfidf_vectorizer.fit_transform(plots)
    print("Vectorization successful. Saving the vectorized tables...")
    # Save the vectorizer and TF-IDF matrix
    with open(OUT_DIR+'/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    with open(OUT_DIR+'/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(df_tfidf, f)


# Custom TF-IDF computation
def create_tf_idf(plots: list):
    print("Starting TF-IDF computation...")
    plots_count = len(plots)  # Total number of plots in collection
    idf_dic = {}
    tf_idf = {}
    alphas = {}

    for plot_idx in range(plots_count):
        for word in plots[plot_idx].split():
            if word not in tf_idf:
                tf_idf[word] = {plot_idx: 0}

            if plot_idx not in tf_idf[word]:
                tf_idf[word][plot_idx] = 0

            tf_idf[word][plot_idx] += 1

        if plot_idx % 5000 == 0 and plot_idx > 0:
            print(f"Processed {plot_idx} plots")
    print("Finished TF computation...")

    for word in tf_idf:
        idf_dic[word] = math.log((1+plots_count)/(1+len(tf_idf[word])))
        for plot_idx in tf_idf[word]:
            tf_idf[word][plot_idx] *= idf_dic[word]
    print("Finished TF-IDF computation...")

    for word in tf_idf:
        for plot_idx, tf_value in tf_idf[word].items():
            if plot_idx not in alphas:
                alphas[plot_idx] = 0.
            alphas[plot_idx] += math.pow(tf_value, 2)
    for plot_idx, tf_value in alphas.items():
        alphas[plot_idx] = math.sqrt(tf_value)
    alphas = dict(sorted(alphas.items(), key=lambda item: item[0]))
    print("Finished alphas computation...")

    # Write to CSV
    print("Outputting IDF table...")
    with open(OUT_DIR + '/idf.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for word, value in idf_dic.items():
            writer.writerow([word] + [str(round(value, 4))])

    print("Outputting TF-IDF table...")
    with open(OUT_DIR + '/tf_idf.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for word, plot_idx in tf_idf.items():
            row = [word]
            for idx, occurrences in plot_idx.items():
                row += [str(idx)] + [str(round(occurrences, 4))]
            writer.writerow(row)

    print("Outputting alphas table...")
    with open(OUT_DIR + '/alphas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for plot_idx, tf_value in alphas.items():
            writer.writerow([plot_idx] + [str(round(tf_value, 4))])


#
if __name__ == '__main__':
    # Load movie data
    print("Load movies dataset")
    df_movies = pd.read_csv(IN_DIR+"/Movie_Plots.csv")
    print("Cleansing movies dataset")
    df_movies.drop(columns=['Origin/Ethnicity', 'Director', 'Cast', 'Genre', 'Wiki Page'], inplace=True)

    # Remove stopwords to of the movie plots
    cleaned_movie_plots = df_movies['Plot'].tolist()
    eliminate_stopwords(cleaned_movie_plots)

    # Create the tf-idf table (using external library or custom computation)
    if use_external_libraries:
        create_tf_idf_ext_lib(cleaned_movie_plots)
    else:
        # Create the tf_idf table
        create_tf_idf(cleaned_movie_plots)

    print("Indexing completed successfully")
