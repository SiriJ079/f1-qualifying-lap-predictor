import pytest
from src.api.predictor_service import PredictorService

@pytest.fixture(scope="module")
def service():
    return PredictorService()

def test_metadata_returns_drivers_and_circuits(service):
    metadata = service.get_metadata()
    assert len(metadata["drivers"]) > 0
    assert len(metadata["circuits"]) > 0
    assert len(metadata["teams"]) > 0

def test_predict_returns_valid_structure(service):
    drivers = service.get_metadata()["drivers"]
    circuits = service.get_metadata()["circuits"]
    result = service.predict(drivers[0], circuits[0], 2026)

    assert "predicted_delta_s" in result
    assert "lower_bound_s" in result
    assert "upper_bound_s" in result
    assert result["lower_bound_s"] <= result["predicted_delta_s"] <= result["upper_bound_s"]

def test_predict_raises_for_unknown_driver(service):
    circuits = service.get_metadata()["circuits"]
    with pytest.raises(ValueError):
        service.predict("ZZZ", circuits[0], 2026)

def test_top_features_present(service):
    drivers = service.get_metadata()["drivers"]
    circuits = service.get_metadata()["circuits"]
    result = service.predict(drivers[0], circuits[0], 2026)
    assert len(result["top_features"]) == 5