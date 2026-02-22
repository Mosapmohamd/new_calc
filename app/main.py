from fastapi import FastAPI

app = FastAPI()

@app.post("/extract-config")
def extract_config(payload: dict):
    title = payload.get("title","")
    description = payload.get("description","")

    return {
        "trims":[
            {"label":"xlt","prob":0.6},
            {"label":"lariat","prob":0.4}
        ],
        "packages":[
            {"label":"tow","prob":0.5}
        ],
        "drivetrains":[
            {"label":"4x4","prob":0.7}
        ],
        "confidence":0.8
    }
