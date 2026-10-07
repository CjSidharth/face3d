"""API-level test for POST /generate, with the TripoSR backend mocked out.

Running the real TripoSR model needs downloaded weights and ~30-90s on MPS/CPU —
too slow and too network-dependent for a unit test. Instead this monkeypatches
`app.main.get_backend` with a fake backend that returns a synthetic bust-like mesh
(reusing the fixture pattern from test_printability.py), so this test validates the
actual FastAPI wiring — request parsing, preprocessing, printability, export, error
mapping — without depending on the reconstruction model itself.
"""
import io

import numpy as np
import PIL.Image
import pytest
import trimesh
from fastapi.testclient import TestClient

import app.main as main_module
from pipeline.preprocess import NoFaceDetectedError


def _make_bust_mesh() -> trimesh.Trimesh:
    head = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    head.apply_translation([0, 0, 12])
    shoulders = trimesh.creation.cylinder(radius=10, height=10, sections=32)
    shoulders.apply_translation([0, 0, 3])
    return trimesh.boolean.union([head, shoulders], engine="manifold")


class _FakeBackend:
    def reconstruct(self, image: PIL.Image.Image) -> trimesh.Trimesh:
        return _make_bust_mesh()


def _fake_image_bytes() -> bytes:
    img = PIL.Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "get_backend", lambda: _FakeBackend())
    return TestClient(main_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_generate_returns_watertight_stl(client, monkeypatch):
    # Skip real face detection/background removal too — this test is about the
    # FastAPI <-> pipeline wiring, not preprocessing correctness (covered
    # separately in test_printability.py's use of preprocess.py).
    monkeypatch.setattr(main_module, "preprocess", lambda image: image)

    files = {"file": ("face.png", _fake_image_bytes(), "image/png")}
    resp = client.post("/generate", files=files, data={"height_mm": "100"})

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] in ("model/stl", "application/octet-stream")
    assert len(resp.content) > 0

    report = main_module.json.loads(resp.headers["x-printability-report"])
    assert report["printability"]["is_watertight"] is True
    assert report["export"]["is_watertight"] is True
    assert report["printability"]["bust_height_mm"] == pytest.approx(100.0, rel=1e-2)


def test_generate_no_face_detected_returns_400(client, monkeypatch):
    def _raise(image):
        raise NoFaceDetectedError("no face here")

    monkeypatch.setattr(main_module, "preprocess", _raise)
    files = {"file": ("face.png", _fake_image_bytes(), "image/png")}
    resp = client.post("/generate", files=files)
    assert resp.status_code == 400
    assert "no face here" in resp.json()["detail"]


def test_generate_rejects_non_image_upload(client):
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    resp = client.post("/generate", files=files)
    assert resp.status_code == 400


def test_generate_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "max_upload_bytes", 10)
    files = {"file": ("face.png", _fake_image_bytes(), "image/png")}
    resp = client.post("/generate", files=files)
    assert resp.status_code == 413
