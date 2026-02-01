import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def full_preprocessing(text):
    # Cleaning
    cleaning = re.sub(r'https?://\S+|www\.\S+', '', text)
    cleaning = re.sub(r'<.*?>', '', cleaning)
    cleaning = re.sub(r'[^a-zA-Z\s]', '', cleaning)

    # Case folding
    case_folding = cleaning.lower()

    # Normalisasi (sederhana)
    normalisasi = case_folding

    # Tokenizing
    tokenize = normalisasi.split()

    # Stopword removal
    stopword_removal = [w for w in tokenize if w not in stop_words]

    # Stemming
    stemming_data = stemmer.stem(" ".join(stopword_removal))

    return {
        "cleaning": cleaning,
        "case_folding": case_folding,
        "normalisasi": normalisasi,
        "tokenize": tokenize,
        "stopword_removal": stopword_removal,
        "stemming_data": stemming_data
    }
