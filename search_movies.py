import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


# Load preprocessed data and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

df_movies = pd.read_csv("preprocessed_movies.csv")

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
def search_plots(query, tfidf_vectorizer, tfidf_matrix, df):
    query_processed = preprocess_text_english(query)
    query_vector = tfidf_vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_titles = df.iloc[sorted_indices]['Title']
    sorted_plots = df.iloc[sorted_indices]['Plot']
    return sorted_titles, sorted_plots, similarities[sorted_indices]

# Get user input for the search query
query = input("Enter your search query: ")

# Perform search
sorted_titles, sorted_plots, sorted_similarities = search_plots(query, tfidf_vectorizer, tfidf_matrix, df_movies)

# Display results
print("Search results for the query:", query)
for title, plot, similarity in zip(sorted_titles.head(10), sorted_plots.head(10), sorted_similarities[:10]):
    print(f"Title: {title}, Similarity: {similarity:.4f}")
    print(f"Plot: {plot}")
    print("------")
