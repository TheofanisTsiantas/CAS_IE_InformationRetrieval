# Necessary packages
import csv
import math
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from CIRS.processor import eliminate_stopwords

# Flag to control if the TF-IDF computation will be done using external libraries
use_external_libraries = False

# Definition of critical directories
SKRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.dirname(SKRIPT_DIR)
OUT_DIR = SKRIPT_DIR + "/IndexFiles.index"


# TF-IDF computation using external libraries
def create_tf_idf_ext_lib(plots: list):
    print("Starting vectorization with external libraries...")
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    df_tfidf = tfidf_vectorizer.fit_transform(plots)
    print("Vectorization successful. Saving the vectorized tables...")
    # Save the vectorizer and TF-IDF matrix
    with open(OUT_DIR + '/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(OUT_DIR + '/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(df_tfidf, f)


# Computation of TF-IDF
def create_tf_idf(plots: list):
    print("Starting TF-IDF computation...")
    plots_count = len(plots)  # Total number of plots in collection
    # For the following dictionary definitions, see also the presentation
    # feature = word
    # document = plot
    idf_dic = {}  # key = feature, value = idf=f(feature)
    tf_idf = {}  # key = feature, value = {key = doc number, value = instances of feature in doc = f(feature, doc)}
    alphas = {}  # alphas dictionary: key = doc number, value = alpha = f(all features in doc)

    # Populate the tf_idf dictionary
    # Iterate over every word (feature) of every plot (document)
    for plot_idx in range(plots_count):
        for word in plots[plot_idx].split():
            # If the feature doesn't exist in the dictionary, add it and initialize it with the current document
            if word not in tf_idf:
                tf_idf[word] = {plot_idx: 0}  # Initialization of the dictionary with the current document

            # If the feature exists in the dictionary, but for another document, add the current document
            if plot_idx not in tf_idf[word]:
                tf_idf[word][plot_idx] = 0  # Addition of the current document to the existing dictionary

            tf_idf[word][plot_idx] += 1  # Increment the feature instances for the current document

        # Output information
        if plot_idx % 5000 == 0 and plot_idx > 0:
            print(f"Processed {plot_idx} plots")
    print("Finished TF computation...")

    # Compute the idf value for each feature and correct the above computed tf_idf structure by
    # the value of idf for each feature
    for word in tf_idf:
        # Computation of idf for each feature
        idf_dic[word] = math.log((1 + plots_count) / (1 + len(tf_idf[word])))
        # Correction of tf_idf for each feature (for all the documents)
        for plot_idx in tf_idf[word]:
            tf_idf[word][plot_idx] *= idf_dic[word]
    # Output information
    print("Finished TF-IDF computation...")

    # Compute the alpha values for each document based on all the feature which exist in the document
    # Loop over all features
    for word in tf_idf:
        # For each feature find the documents which reference the feature
        for plot_idx, tf_value in tf_idf[word].items():
            # If the document has not been added to the final dictionary initialize it
            if plot_idx not in alphas:
                alphas[plot_idx] = 0.
            # Add the contribution of the tf_idf value of the current document
            alphas[plot_idx] += math.pow(tf_value, 2)
    # Compute the square root after the contribution of all documents has been considered
    for plot_idx, tf_value in alphas.items():
        alphas[plot_idx] = math.sqrt(tf_value)
    # Sort the alphas based on the document number
    alphas = dict(sorted(alphas.items(), key=lambda item: item[0]))
    # Output information
    print("Finished alphas computation...")

    # Write to CSVs
    # CSV output for IDF table
    print("Outputting IDF table...")
    # add encoding='utf-8' as param if using Windows system
    with open(OUT_DIR + '/idf.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for word, value in idf_dic.items():
            writer.writerow([word] + [str(round(value, 4))])
    # CSV output for TF-IDF table
    print("Outputting TF-IDF table...")
    # add encoding='utf-8' as param if using Windows system
    with open(OUT_DIR + '/tf_idf.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for word, plot_idx in tf_idf.items():
            row = [word]
            for idx, occurrences in plot_idx.items():
                row += [str(idx)] + [str(round(occurrences, 4))]
            writer.writerow(row)
    # CSV output for alphas table
    print("Outputting alphas table...")
    # add encoding='utf-8' as param if using Windows system
    with open(OUT_DIR + '/alphas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for plot_idx, tf_value in alphas.items():
            writer.writerow([plot_idx] + [str(round(tf_value, 4))])


if __name__ == '__main__':
    # Load and cleanse movie data
    print("Load movies dataset")
    df_movies = pd.read_csv(IN_DIR + "/Movie_Plots.csv")
    print("Cleansing movies dataset")
    df_movies.drop(columns=['Origin/Ethnicity', 'Director', 'Cast', 'Genre', 'Wiki Page'], inplace=True)

    # Remove stopwords to of the movie plots
    cleaned_movie_plots = df_movies['Plot'].tolist()
    eliminate_stopwords(cleaned_movie_plots)

    # Create the tf-idf table (using external library or custom computation)
    if use_external_libraries:
        create_tf_idf_ext_lib(cleaned_movie_plots)
    else:
        # Compute the tf_idf table
        create_tf_idf(cleaned_movie_plots)
    print("Indexing completed successfully")
