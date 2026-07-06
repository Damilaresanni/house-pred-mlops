from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

import pandas as pd

app=FastAPI(
    title= "A House Price prediction API",
    description= (
        "This is a machine learning project that utilizes house data in"
    "india to train a model to provide intelligent system using an API endpoint"
    ),
    version="0.1.0",
    
    contact={
        'name':"Abdulhakeem Sanni",
        'email':"sanniabdulhakeem5@gmail.com"
    }
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_credentials=True,
    allow_methods=['*']
    
)


model = joblib.load("/home/fidisroxy/development/mlops/house-pred-mlops/models/model.pkl")

class HouseData(BaseModel):
    
    location: str
    bathroom: int
    carpet_area: int
    status: str
    balcony: int

@app.get("/")
def health_check():
    return{"status":"Healthy"}

@app.post("/predict")
def predict(data:HouseData):
    
    input_df = pd.DataFrame([data.model_dump()])
    
    prediction = model.predict(input_df)
    
    return{"predicted_price":float(prediction[0])}