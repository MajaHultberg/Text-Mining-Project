import pandas as pd
import spacy
from langdetect import detect, DetectorFactory
nlp = spacy.load("en_core_web_sm")

# Create dataframe
df = pd.read_csv('all_songs_data_processed.csv')
# Remove unnecessary info
df = df.drop(columns=['Album','Album URL','Featured Artists', 'Media', 'Release Date', 'Song URL', 'Writers', 'Verbs','Nouns', 'Adverbs', 'Corpus', 'Word Counts', 'Unique Word Counts'])
df = df[(df.Year != 1959) & (df.Year < 2020)]

# Remove non-english songs
DetectorFactory.seed = 0
def detect_language(text):
    try:
        return detect(text)
    except:
        return None
df['Language'] = df['Lyrics'].apply(detect_language)
df = df[df['Language'] == 'en']

# Add decade tags
decades = []
for item in df.Year.tolist():
    if item < 1970:
        decades.append('60s')
    if 1970 <= item < 1980:
        decades.append('70s')
    if 1980 <= item < 1990:
        decades.append('80s')
    if 1990 <= item < 2000:
        decades.append('90s')
    if 2000 <= item < 2010:
        decades.append('00s')
    if item >= 2010:
        decades.append('10s')
df['Decade'] = decades

# Tokenize, lemmatize, remove stopwords and punctuation from lyrics
def tokenize_and_lemmatize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
df['Lemmatized Lyrics'] = df['Lyrics'].apply(tokenize_and_lemmatize)

df.to_csv('preprocessed_data.csv')