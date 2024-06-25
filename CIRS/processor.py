import spacy
from nltk.corpus import stopwords

# Definition of stopwords
STOPS_NLTK = stopwords.words('english')

# Definition of spaCy model (load)
nlp = spacy.load('en_core_web_sm')


# Method to eliminate all unnecessary words from the list of plots
def eliminate_stopwords(plots: list):
    print("Starting removal of stopwords...")
    # For every plot in the list of plots
    for plot_idx in range(0, len(plots)):
        # Create a list of split words
        plot_words = plots[plot_idx].split()
        # Define a list which will contain the necessary plot words
        cleaned_words = []
        for word in plot_words:
            # Check in the standard NLTK-package
            if word.lower() not in STOPS_NLTK:
                cleaned_words.append(word.lower())
        # Replace the plot with the necessary words only
        plots[plot_idx] = ' '.join(cleaned_words)

        if plot_idx % 5000 == 0 and plot_idx > 0:
            print(f"Processed {plot_idx} plots")
    print("Removal of stopwords finished...")
