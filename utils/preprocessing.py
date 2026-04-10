import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

nltk.download("stopwords")

# =========================
# STOPWORDS (IDENTIK DATASET)
# =========================
stop_words = set(stopwords.words("indonesian"))

negation_words = {'tidak','bukan','belum','kurang','jangan'}
stop_words = stop_words - negation_words

custom_stopwords = {
    'dong','nih','deh','ya','yaa','kak','hehe','tuh','hai','oh','iya',
    'wkwk','wk','sih','aja','lah','loh','nya','pun',
    'mu','ku','mah','min','admin',
    'halo','haloo','pls','please',
    'amp','ampulnya','oomfs','wtb','it','nder','ann',
    'ombb','www','st','mjb','aaaa','t','wal','oot','dll',
    'wkwkw','wkwkwk','wkwkwkwk','wkwkwkwkkw',
    'wkwkwkwkwk','wkwkkwkw','wkwkwkwkwk',
    'hehehehe','hehehhe',
    'jir','huft','hahahaha','xixi','hiks','huhu',
    'gweh','cowoku','abang','mamih2',
    'zonauang','zonaba','wts','fomo',
    'indomaret','watsons','halodoc',
    'cuman','doang','kayaknya','seperti','mirip',
    'siang','malam','pagi','sore',
    'kali','minggu','bulan','tahun',
    'nderr','btw','fyi','an',
    'oiya','hmmm','woi','hehehe','hihiw','yarabb',
    'plis','ajahh','dikiit',
    'aku','i',"i've",'me','my',
    'temenku','tante','mamah','anak',
    'gadis','bocil','cowok','t','nderrrrrrr','nderrr'
}

stop_words.update(custom_stopwords)

important_words = {
    'cocok','kering','jerawat','lembap',
    'wajah','kulit','pakai','banget'
}

stop_words = stop_words - important_words


# =========================
# STEMMER
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

protected_words = {
    "facial","wash","cleanser",
    "foam","gel","gentle",
    "wardah","garnier","cetaphil",
    "glad2glow","hadalabo","skintific","cosrx",
    "glowandlovely",
    "wajah","kulit","jerawat"
}


# =========================
# LOAD KAMUS (IDENTIK)
# =========================
kamus_alay_df = pd.read_csv("kamus_alay.csv")
kamus_alay = dict(zip(kamus_alay_df['slang'], kamus_alay_df['formal']))

kamus_baku_df = pd.read_excel("kamuskatabaku.xlsx")
kamus_kata_baku = dict(zip(kamus_baku_df['tidak_baku'], kamus_baku_df['kata_baku']))

kamus_skincare = {

    "fw": "facial wash",
    "fwnya": "facial wash",
    "facewash": "facial wash",
    "face wash": "facial wash",
    "face_wash": "facial wash",
    "faciawash": "facial wash",
    "facialwash": "facial wash",

    "ss": "sunscreen",
    "sun screen": "sunscreen",
    "mw": "micellar water",
    "mc": "milk cleanser",

    "g2g": "glad2glow",
    "glad2 glow": "glad2glow",

    "hada labo": "hadalabo",
    "hada-labo": "hadalabo",

    "crosx": "cosrx",
    "cos rx": "cosrx",

    "glow & lovely": "glowandlovely",
    "glow n lovely": "glowandlovely",
    "glow and lovely": "glowandlovely",

    "exfo": "exfoliating",
    "ampul": "ampoule",
    "ampule": "ampoule",

    "mois": "moisturizer",
    "moist": "moisturizer",
    "moistnya": "moisturizer",
    "moisturiser": "moisturizer",

    "ijo": "hijau",
    "ijoo": "hijau",
    "ijooo": "hijau",

    "ketarik": "terasa tertarik",
    "kesettt": "kesat",
    "licinnn": "licin",
    "lembab": "lembap",

    "jerawatan": "berjerawat",
    "seger": "segar",

    "muka": "wajah",
    "mukaku": "wajah",
    "gue": "aku",
    "gw": "aku",
    "gua": "aku"
}

kamus_normalisasi = {}
kamus_normalisasi.update(kamus_alay)
kamus_normalisasi.update(kamus_kata_baku)
kamus_normalisasi.update(kamus_skincare)


# =========================
# NORMALISASI
# =========================
def normalize_words(text, kamus):

    for key in kamus:
        if " " in key:
            text = text.replace(key, kamus[key])

    words = text.split()
    words = [kamus.get(w, w) for w in words]

    return " ".join(words)


# =========================
# FULL PREPROCESSING
# =========================
def preprocessing_data(text):

    # ===== CLEANING =====
    cleaning = re.sub(r'https?://\S+|www\.\S+', '', text)
    cleaning = re.sub(r'<.*?>', '', cleaning)
    cleaning = re.sub(r'@\w+', '', cleaning)
    cleaning = re.sub(r'\bRT\b', '', cleaning)

    cleaning = re.sub(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        "]+", '', cleaning
    )

    # ===== PROTEKSI BRAND =====
    cleaning = re.sub(r'glad2glow', 'PROTEXGGLONG', cleaning, flags=re.IGNORECASE)
    cleaning = re.sub(r'\bg2g\b', 'PROTEXGGSHORT', cleaning, flags=re.IGNORECASE)

    # ===== HAPUS ANGKA =====
    cleaning = re.sub(r'\b\d+(st|nd|rd|th)\b', '', cleaning, flags=re.IGNORECASE)
    cleaning = re.sub(r'(?i)rp\s?\d+([.,]\d+)*', '', cleaning)
    cleaning = re.sub(r'\d+', '', cleaning)

    # ===== KEMBALIKAN BRAND =====
    cleaning = re.sub(r'PROTEXGGLONG', 'glad2glow', cleaning)
    cleaning = re.sub(r'PROTEXGGSHORT', 'g2g', cleaning)

    # ===== SIMBOL & SPASI =====
    cleaning = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaning)
    cleaning = re.sub(r'\s+', ' ', cleaning).strip()

    # ===== CASE FOLDING =====
    case_folding = cleaning.lower().strip()

    # ===== NORMALISASI =====
    normalisasi = normalize_words(case_folding, kamus_normalisasi)

    # ===== TOKENIZE =====
    tokenize = normalisasi.split()

    # ===== STOPWORD =====
    stopword_removal = [w for w in tokenize if w not in stop_words]

    # ===== STEMMING =====
    result = []
    for word in stopword_removal:
        if word in protected_words:
            result.append(word)
        else:
            result.append(stemmer.stem(word))

    stemming_data = ' '.join(result)

    return {
        "cleaning": cleaning,
        "case_folding": case_folding,
        "normalisasi": normalisasi,
        "tokenize": tokenize,
        "stopword_removal": stopword_removal,
        "stemming_data": stemming_data
    }