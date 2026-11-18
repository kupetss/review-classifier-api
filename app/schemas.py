from pydantic import BaseModel

class ReviewRequest(BaseModel):
    text: str

class PredicionResponse(BaseModel):
    sentiment: str
    probability: float

