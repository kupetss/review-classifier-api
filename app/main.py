from fastapi import FastAPI

from app.model import predict
from app.schemas import PredicionResponse, ReviewRequest

app = FastAPI(title="Review Classifier API")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredicionResponse)
def predict_sentiment(request: ReviewRequest):
    sentiment, probability = predict(request.text)
    return PredicionResponse(sentiment=sentiment, probability=probability)