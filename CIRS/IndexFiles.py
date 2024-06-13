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
    plots_count = 10 #len(plots)  # Total number of plots in collection
    test_dir = {}

    for plot_idx in range(plots_count):
        for word in plots[plot_idx].split():
            if word not in test_dir:
                test_dir[word] = {plot_idx: 0}

            if plot_idx not in test_dir[word]:
                test_dir[word][plot_idx] = 0

            test_dir[word][plot_idx] += 1

        if plot_idx % 5000 == 0 and plot_idx > 0:
            print(f"Processed {plot_idx} plots")
    print("Finished TF-IDF computation...")

    # Write to CSV
    with open(OUT_DIR + '/output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        #
        for word, plot_idx in test_dir.items():
            row = [word]
            for idx, occurrences in plot_idx.items():
                row += str(idx) + str(occurrences)
            writer.writerow(row)

    # Open the CSV file
    with open(OUT_DIR + '/output.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        test_dir2 = {}
        # Read the header
        for row in csv_reader:
            data = row
            test_dir2[data[0]] = {data[1]: data[2]}
            for data_idx in range(3, len(data), 2):
                test_dir2[data[0]][data[data_idx]] = [data[data_idx+1]]

    t = 1
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
