import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

nltk.download("stopwords")

# Stopword Indonesia + jangan hapus negasi
stop_words = stopwords.words("indonesian")
negation_words = ['tidak', 'bukan', 'kurang', 'belum']
stop_words = [w for w in stop_words if w not in negation_words]

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Kamus kata baku
kamus_df = pd.read_excel("kamuskatabaku.xlsx")
kamus_tidak_baku = dict(zip(kamus_df['tidak_baku'], kamus_df['kata_baku']))

# ============================
# FULL PREPROCESSING
# ============================
def preprocessing_data(text):

    # 1. Cleaning
    cleaning = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL
    cleaning = re.sub(r'<.*?>', '', cleaning)             # HTML
    cleaning = re.sub(r'@\w+', '', cleaning)              # Mention

    # Hapus ordinal seperti 1st, 2nd, 3rd, 4th
    cleaning = re.sub(r'\b\d+(st|nd|rd|th)\b', '', cleaning, flags=re.IGNORECASE)

    # Hapus harga format Rp (Rp20.000, rp 15,000, RP10000)
    cleaning = re.sub(r'(?i)rp\s?\d+([.,]\d+)*', '', cleaning)

    # Hapus angka biasa
    cleaning = re.sub(r'\d+', '', cleaning)

    # Hapus simbol selain huruf
    cleaning = re.sub(r'[^a-zA-Z\s]', ' ', cleaning)

    # Rapikan spasi berlebih
    cleaning = re.sub(r'\s+', ' ', cleaning).strip()

    # 2. Case Folding
    case_folding = cleaning.lower()

    # 3. Normalisasi (kamus kata baku)
    words = case_folding.split()
    normalized_words = [kamus_tidak_baku.get(w, w) for w in words]
    normalisasi = " ".join(normalized_words)

    # 4. Tokenizing
    tokenize = normalisasi.split()

    # 5. Stopword Removal
    stopword_removal = [w for w in tokenize if w not in stop_words]

    # 6. Stemming
    stemming_data = " ".join([stemmer.stem(w) for w in stopword_removal])

    return {
        "cleaning": cleaning,
        "case_folding": case_folding,
        "normalisasi": normalisasi,
        "tokenize": tokenize,
        "stopword_removal": stopword_removal,
        "stemming_data": stemming_data
    }
