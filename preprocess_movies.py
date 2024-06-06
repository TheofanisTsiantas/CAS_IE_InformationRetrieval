import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Load data
df_movies = pd.read_csv("Movie_Plots.csv")

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

# Apply preprocessing to the 'Plot' column
df_movies['preprocessed_text'] = df_movies['Plot'].apply(preprocess_text_english)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
df_tfidf = tfidf_vectorizer.fit_transform(df_movies['preprocessed_text'])

# Save the vectorizer and TF-IDF matrix
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(df_tfidf, f)

# Save the processed DataFrame
df_movies.to_csv("preprocessed_movies.csv", index=False)
