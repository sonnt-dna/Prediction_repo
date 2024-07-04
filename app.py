from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

# Define the request body
class PredictionRequest(BaseModel):
    DEPTH: float
    CALI: float
    DT: float
    GR: float
    LLD: float
    LLS: float
    NPHI: float
    PHIE: float
    RHOB: float
    SW: float

# Define the response body
class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Prepare the data for prediction
        data = np.array([[request.DEPTH, request.CALI, request.DT, request.GR, request.LLD, request.LLS, request.NPHI, request.PHIE, request.RHOB, request.SW]])

        # Make the prediction
        prediction = model.predict(data)

        return PredictionResponse(prediction=prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Other endpoints and logic
