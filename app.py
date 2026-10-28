
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("ca2mlops.joblib")

class PredictionInput(BaseModel):
    Year: int
    LocationAbbr: str
    LocationDesc: str
    GeographicLevel: str
    StratificationCategory1: str
    Stratification1: str
    StratificationCategory2: str
    Stratification2: str

@app.get("/")
def home():
    return {"message": "Stroke Mortality Prediction API"}

@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    return {
        "predicted_stroke_mortality_rate": round(float(prediction), 2)
    }
