import joblib

def load_model():
    saved = joblib.load("model_sentimen_logistic_regression.pkl")
    return saved['model'], saved['tfidf']
