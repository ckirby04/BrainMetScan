"""
API client for connecting the web viewer to the BrainMetScan v1.23 backend.
Enables the Gradio demo to run predictions via the REST API instead of
loading models locally.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class BrainMetScanAPIClient:
    """Client for the BrainMetScan REST API (v1.23)."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def health(self) -> Dict:
        """Check API health status."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def is_available(self) -> bool:
        """Check if the API is reachable and has models loaded."""
        health = self.health()
        return health.get("status") == "healthy"

    def list_models(self) -> List[Dict]:
        """List available models on the server."""
        r = self.session.get(f"{self.base_url}/models", timeout=5)
        r.raise_for_status()
        return r.json()

    def predict(
        self,
        nifti_paths: Dict[str, str],
        threshold: float = 0.5,
        use_tta: bool = False,
        case_id: str = "",
        generate_report: bool = False,
    ) -> Dict:
        """
        Run prediction via the API.

        Args:
            nifti_paths: Dict mapping sequence names to NIfTI file paths
                         e.g. {"t1_pre": "/path/to/t1_pre.nii.gz", ...}
            threshold: Detection threshold
            use_tta: Whether to use test-time augmentation
            case_id: Optional case identifier
            generate_report: Whether to generate RAG report

        Returns:
            API prediction response dict
        """
        files = []
        for seq_name, path in nifti_paths.items():
            path = Path(path)
            files.append(
                ("files", (f"{seq_name}.nii.gz", open(path, "rb"), "application/gzip"))
            )

        data = {
            "input_format": "nifti",
            "threshold": str(threshold),
            "use_tta": str(use_tta).lower(),
            "case_id": case_id,
            "generate_report": str(generate_report).lower(),
        }

        try:
            r = self.session.post(
                f"{self.base_url}/predict",
                files=files,
                data=data,
                timeout=300,  # 5 min for large volumes
            )
            r.raise_for_status()
            return r.json()
        finally:
            for _, (_, f, _) in files:
                f.close()

    def download_mask(self, job_id: str, output_format: str = "nifti") -> bytes:
        """Download segmentation mask for a completed prediction."""
        r = self.session.get(
            f"{self.base_url}/predict/{job_id}/mask",
            params={"output_format": output_format},
            timeout=30,
        )
        r.raise_for_status()
        return r.content

    def download_report(self, job_id: str) -> bytes:
        """Download PDF report for a completed prediction."""
        r = self.session.get(
            f"{self.base_url}/predict/{job_id}/report",
            timeout=30,
        )
        r.raise_for_status()
        return r.content

    def compare(
        self,
        baseline_paths: Dict[str, str],
        followup_paths: Dict[str, str],
        threshold: float = 0.5,
    ) -> Dict:
        """Run longitudinal comparison between two timepoints."""
        files = []
        for seq_name, path in baseline_paths.items():
            files.append(
                ("baseline_files", (f"{seq_name}.nii.gz", open(Path(path), "rb"), "application/gzip"))
            )
        for seq_name, path in followup_paths.items():
            files.append(
                ("followup_files", (f"{seq_name}.nii.gz", open(Path(path), "rb"), "application/gzip"))
            )

        data = {"threshold": str(threshold)}

        try:
            r = self.session.post(
                f"{self.base_url}/compare",
                files=files,
                data=data,
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        finally:
            for _, (_, f, _) in files:
                f.close()

    def get_stats(self, days: int = 30) -> Dict:
        """Get usage statistics from the admin endpoint."""
        r = self.session.get(
            f"{self.base_url}/admin/stats",
            params={"days": days},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()


def create_client_from_env() -> BrainMetScanAPIClient:
    """Create an API client from environment variables."""
    import os

    base_url = os.environ.get("BRAINMETSCAN_API_URL", "http://localhost:8000")
    api_key = os.environ.get("BRAINMETSCAN_API_KEY", "")
    return BrainMetScanAPIClient(base_url=base_url, api_key=api_key or None)
