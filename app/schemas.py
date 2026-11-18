from pydantic import BaseModel, ConfigDict, Field

class ReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=10000)

class PredictionResponse(BaseModel):
    sentiment: str
    probability: float

