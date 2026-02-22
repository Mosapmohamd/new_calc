# routes/predict_price.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Req(BaseModel):
    year: int
    make: str
    model: str
    mileage: int
    trims: List[dict]
    regime: dict

class Res(BaseModel):
    p10: float
    p50: float
    p90: float
    confidence: float

BASE_PRICE = {
    ("ford","f150"): 30000
}

@router.post("/predict-price", response_model=Res)
def predict(req: Req):
    base = BASE_PRICE.get((req.make,req.model),25000)

    adj = 0
    for t in req.trims:
        if t["label"]=="lariat":
            adj += 4000*t["prob"]
        if t["label"]=="platinum":
            adj += 8000*t["prob"]

    price = base + adj - (req.mileage*0.05)

    return {
        "p10": price*0.9,
        "p50": price,
        "p90": price*1.1,
        "confidence": 0.7
    }
