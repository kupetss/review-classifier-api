from pathlib import Path
import joblib
from ml.preprocess import clean_text

MODEL_PATH = Path("model/model.joblib")
model = joblib.load(MODEL_PATH)

def predict(text: str) -> tuple[str, float]:
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0][1]

    if proba >= 0.5:
        return "positive", proba
    return "negative", 1 - proba