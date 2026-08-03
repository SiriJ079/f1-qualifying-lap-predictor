from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictionRequest, PredictionResponse, CompareRequest, CompareResponse, MetadataResponse
from src.api.predictor_service import PredictorService

app = FastAPI(title="F1 Qualifying Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PredictorService()

@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    return service.get_metadata()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return service.predict(request.driver, request.circuit, request.year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    predictions = []
    for driver in request.drivers:
        try:
            predictions.append(service.predict(driver, request.circuit, request.year))
        except ValueError:
            continue
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "ok"}