import spacy
from nltk.corpus import stopwords

# Definition of stopwords
STOPS_NLTK = stopwords.words('english')

# Definition of spaCy model (load)
nlp = spacy.load('en_core_web_sm')


def eliminate_stopwords(plots: list):
    print("Starting removal of stopwords...")
    for plot_idx in range(0, len(plots)):
        plot_words = plots[plot_idx].split()
        cleaned_words = []
        for word in plot_words:
            if word.lower() not in STOPS_NLTK:
                cleaned_words.append(word.lower())
        plots[plot_idx] = ' '.join(cleaned_words)

        if plot_idx % 5000 == 0 and plot_idx > 0:
            print(f"Processed {plot_idx} plots")
    print("Removal of stopwords finished...")
