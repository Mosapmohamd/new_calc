# routes/extract_config.py
import re
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class Req(BaseModel):
    title: str = ""
    description: str = ""

class Prob(BaseModel):
    label: str
    prob: float

class Res(BaseModel):
    trims: List[Prob]
    packages: List[Prob]
    drivetrains: List[Prob]
    confidence: float

def clean(t: str):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

TRIM = {
    "xl": "xl",
    "xlt": "xlt",
    "lariat": "lariat",
    "denali": "denali",
    "ltz": "ltz",
    "platinum": "platinum"
}

PKG = {
    "fx4": "fx4",
    "z71": "z71",
    "tow": "tow",
    "off road": "offroad"
}

DRV = {
    "4x4": "4x4",
    "4wd": "4x4",
    "awd": "awd",
    "2wd": "2wd"
}

def rule_probs(text: str, mapping: Dict[str,str]):
    scores = {}
    for k,v in mapping.items():
        if k in text:
            scores[v] = scores.get(v,0)+1
    s = sum(scores.values())
    if s==0:
        return {}
    return {k:v/s for k,v in scores.items()}

def to_list(d):
    return [{"label":k,"prob":float(v)} for k,v in d.items()]

@router.post("/extract-config", response_model=Res)
def extract(req: Req):
    text = clean(req.title + " " + req.description)

    trims = rule_probs(text, TRIM)
    pkgs = rule_probs(text, PKG)
    drv = rule_probs(text, DRV)

    confidence = min(1.0, len(text.split())/40)

    return {
        "trims": to_list(trims),
        "packages": to_list(pkgs),
        "drivetrains": to_list(drv),
        "confidence": confidence
    }
