from fastapi import FastAPI
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

model = joblib.load("models/model.pkl")

class HouseData(BaseModel):
    pass

@app.get("/")
def health_check():
    return{"status":"Healthy"}

@app.post("/predict")
def prredict(data:HouseData):
    
    input_df = pd.DataFrame([data.model_dump()])
    
    prediction = model.predict(input_df)
    
    return{"predicted_price":float(prediction[0])}