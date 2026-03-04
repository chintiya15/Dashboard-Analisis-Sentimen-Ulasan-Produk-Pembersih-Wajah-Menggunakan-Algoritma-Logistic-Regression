import joblib

def load_model():
    model = joblib.load("model_sentimen_logistic_regression.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf
