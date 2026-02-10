import joblib

def load_model():
    model = joblib.load("model_sentimen_smoote.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf
