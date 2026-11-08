# app.py - a small web API that answers car-price questions using champion.pkl
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Car Price API")

# Load the model ONCE when the API starts (not on every request)
model = joblib.load("champion.pkl")

# The shape of a request. FastAPI rejects anything that doesn't match.
class Car(BaseModel):
    year: int = Field(ge=1990, le=2026, description="Model year")
    cc: int = Field(ge=600, le=6000, description="Engine size in cc")
    mileage: int = Field(ge=0, le=500000, description="Kilometres driven")
    imported: int = Field(ge=0, le=1, description="1 = imported, 0 = local")
    automatic: int = Field(ge=0, le=1, description="1 = automatic, 0 = manual")

# Door 1: "are you alive?"
@app.get("/health")
def health():
    return {"status": "ok"}

# Door 2: the real job
@app.post("/predict")
def predict(car: Car):
    # Column order must match the order used in train.py
    data = pd.DataFrame([{
        "year": car.year,
        "cc": car.cc,
        "mileage": car.mileage,
        "imported": car.imported,
        "automatic": car.automatic,
    }])
    price = model.predict(data)[0]
    return {"price_lakh": round(float(price), 1)}