from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    driver: str
    circuit: str
    year: int = 2026

class PredictionResponse(BaseModel):
    driver: str
    circuit: str
    predicted_delta_s: float
    lower_bound_s: float
    upper_bound_s: float
    top_features: List[dict]

class CompareRequest(BaseModel):
    drivers: List[str]
    circuit: str
    year: int = 2026

class CompareResponse(BaseModel):
    predictions: List[PredictionResponse]

class MetadataResponse(BaseModel):
    drivers: List[str]
    circuits: List[str]
    teams: List[str]