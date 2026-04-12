from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the src folder into python path to import the inference code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from inference import predict

app = FastAPI(title="Mental Health API")

# Setup CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    gender: int
    age: float
    cgpa: float
    marital_status: int
    anxiety: int
    panic_attack: int
    treatment: int

@app.post("/predict")
def predict_endpoint(request: InferenceRequest):
    input_data = request.dict()
    result = predict(input_data)
    return result

@app.get("/health")
def health_check():
    return {"status": "ok"}
