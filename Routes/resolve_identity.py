# routes/resolve_identity.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Req(BaseModel):
    year: int
    make: str
    model: str

class Prob(BaseModel):
    label: str
    prob: float

class Res(BaseModel):
    identity_trims: List[Prob]
    identity_packages: List[Prob]
    confidence: float

# placeholder priors
PRIORS = {
    ("ford","f150"): {
        "trims": {"xl":0.2,"xlt":0.4,"lariat":0.25,"platinum":0.15},
        "packages": {"tow":0.5,"fx4":0.3}
    }
}

def to_list(d):
    return [{"label":k,"prob":v} for k,v in d.items()]

@router.post("/resolve-identity", response_model=Res)
def resolve(req: Req):
    key = (req.make.lower(), req.model.lower())
    data = PRIORS.get(key, {"trims":{},"packages":{}})

    return {
        "identity_trims": to_list(data["trims"]),
        "identity_packages": to_list(data["packages"]),
        "confidence": 0.6 if data["trims"] else 0.2
    }
