# routes/classify_regime.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Req(BaseModel):
    year: int
    make: str
    model: str
    mileage: int
    trims: List[dict] = []

class Res(BaseModel):
    regime_probs: dict

@router.post("/classify-regime", response_model=Res)
def classify(req: Req):
    work = 0.5
    lifestyle = 0.3
    luxury = 0.2

    for t in req.trims:
        if t["label"] in ["xl","wt"]:
            work += 0.2
        if t["label"] in ["platinum","denali"]:
            luxury += 0.3

    total = work+lifestyle+luxury

    return {
        "regime_probs":{
            "work": work/total,
            "lifestyle": lifestyle/total,
            "luxury": luxury/total
        }
    }
