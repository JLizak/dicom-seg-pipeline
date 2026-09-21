"""Klient REST Orthanca (hierarchia Patient → Study → Series → Instance).

Dane logowania z env (`ORTHANC_URL`, `ORTHANC_USER`, `ORTHANC_PASSWORD`),
domyślnie lokalny Orthanc z docker-compose.
"""

import os

import requests


class OrthancClient:
    def __init__(self, url: str | None = None, user: str | None = None, password: str | None = None):
        self.url = (url or os.getenv("ORTHANC_URL", "http://localhost:8042")).rstrip("/")
        self.session = requests.Session()
        self.session.auth = (user or os.getenv("ORTHANC_USER", "dev"), password or os.getenv("ORTHANC_PASSWORD", "dev"))

    def get(self, path: str) -> dict | list:
        resp = self.session.get(f"{self.url}/{path.lstrip('/')}")
        resp.raise_for_status()
        return resp.json()

    def patients(self) -> list[str]:
        return self.get("/patients")

    def patient(self, patient_id: str) -> dict:
        return self.get(f"/patients/{patient_id}")

    def study(self, study_id: str) -> dict:
        return self.get(f"/studies/{study_id}")

    def series(self, series_id: str) -> dict:
        return self.get(f"/series/{series_id}")

    def instance(self, instance_id: str) -> dict:
        return self.get(f"/instances/{instance_id}")

    def instance_file(self, instance_id: str) -> bytes:
        """Surowy plik DICOM (do pydicom.dcmread(BytesIO(...)))."""
        resp = self.session.get(f"{self.url}/instances/{instance_id}/file")
        resp.raise_for_status()
        return resp.content
