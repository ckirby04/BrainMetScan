"""Integration tests for FastAPI endpoints using TestClient."""

import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip if SimpleITK not installed (needed by server imports)
SimpleITK = pytest.importorskip("SimpleITK")
nib = pytest.importorskip("nibabel")


@pytest.fixture(scope="module")
def client():
    """Create TestClient with mocked ensemble to avoid loading real models."""
    mock_ensemble = MagicMock()
    mock_ensemble.models = []
    mock_ensemble.names = []

    with patch("src.segmentation.ensemble.SmartEnsemble.from_config", return_value=mock_ensemble):
        from fastapi.testclient import TestClient
        from src.api.server import app

        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "models_loaded" in data

    def test_health_shows_no_models(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["models_loaded"] == 0


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestPredictEndpoint:
    def test_predict_no_models_returns_503(self, client):
        """Prediction should fail gracefully when no models are loaded."""
        files = []
        for seq in ["t1_pre", "t1_gd", "flair", "t2"]:
            arr = np.zeros((8, 8, 8), dtype=np.float32)
            nii_img = nib.Nifti1Image(arr, np.eye(4))

            buf = io.BytesIO()
            file_map = nib.FileHolder(fileobj=buf)
            nii_img.to_file_map({"image": file_map, "header": file_map})
            buf.seek(0)

            files.append(("files", (f"{seq}.nii.gz", buf, "application/gzip")))

        response = client.post(
            "/predict",
            files=files,
            data={"input_format": "nifti", "threshold": "0.5"},
        )
        assert response.status_code == 503
