# predict.py - uses the saved model to price one car (no training here)
import joblib
import pandas as pd

model = joblib.load("model.pkl")

# A 2017 Honda City: 1300 cc, 55,000 km, local assembly, manual
car = pd.DataFrame([{
    "year": 2017,
    "cc": 1300,
    "mileage": 55000,
    "imported": 0,
    "automatic": 0,
}])

price = model.predict(car)[0]
print(f"Predicted price: {price:.1f} lakh")