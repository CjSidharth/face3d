"""FastAPI app: a single endpoint that turns an uploaded face photo into a
printable STL bust.

POST /generate
    multipart form:
        file: image/jpeg or image/png, a frontal face photo
        height_mm: optional float form field — target bust height in mm
                   (default: settings.default_height_mm). This is the bust's own
                   height; the pedestal adds settings.pedestal_height_mm on top.
    response:
        200: binary STL (Content-Type: model/stl). The printability + export
             report is serialized as JSON in the `X-Printability-Report` header
             (curl -i / -D - to see it).
        4xx: JSON {"detail": "..."} for a problem with the input (bad image, no
             face detected, ambiguous silhouette after background removal).
        500: JSON {"detail": "..."} if the pipeline itself failed to produce a
             verified-watertight STL — this is a server-side guarantee failure,
             not a user input problem.

GET /health — liveness + which device (mps/cpu) reconstruction is running on.
"""
import io
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import PIL.Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from pipeline.export import ExportError, export_stl
from pipeline.preprocess import NoFaceDetectedError, preprocess
from pipeline.printability import PrintabilityError, make_printable
from pipeline.reconstruct.triposr_backend import ReconstructionError, TripoSRBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face3d")

# TripoSR weights are ~hundreds of MB and take real time to load; construct the
# backend once (at startup, see `lifespan` below) and reuse it across requests
# rather than loading it per-request.
_backend: Optional[TripoSRBackend] = None


def get_backend() -> TripoSRBackend:
    global _backend
    if _backend is None:
        _backend = TripoSRBackend()
    return _backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TripoSR backend on device=%s ...", settings.device)
    get_backend().model  # touch the lazy `model` property to force the load now
    logger.info("TripoSR backend ready.")
    yield


app = FastAPI(
    title="face3d",
    version="0.1.0",
    description="Turn a single frontal face photo into a 3D-printable STL bust.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": settings.device}


@app.post("/generate")
async def generate(
    file: UploadFile = File(..., description="Frontal face photo (JPEG/PNG)"),
    height_mm: Optional[float] = Form(
        None, description="Target bust height in mm (default: settings.default_height_mm)"
    ),
):
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"Upload too large: {len(content)} bytes (max {settings.max_upload_bytes})."
        )

    try:
        image = PIL.Image.open(io.BytesIO(content))
        image.load()
    except Exception as exc:
        raise HTTPException(400, f"Could not read uploaded file as an image: {exc}") from exc

    try:
        prepped = preprocess(image)
    except NoFaceDetectedError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        raw_mesh = get_backend().reconstruct(prepped)
    except ReconstructionError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        mesh, printability_report = make_printable(raw_mesh, height_mm=height_mm)
    except PrintabilityError as exc:
        raise HTTPException(422, str(exc)) from exc

    output_path = settings.outputs_dir / f"bust_{uuid.uuid4().hex}.stl"
    try:
        export_report = export_stl(mesh, output_path)
    except ExportError as exc:
        # The pipeline promises a verified-watertight STL; failing that check here
        # is our own guarantee breaking, not a bad user input, hence 500.
        raise HTTPException(500, str(exc)) from exc

    report = {"printability": asdict(printability_report), "export": asdict(export_report)}
    return FileResponse(
        output_path,
        media_type="model/stl",
        filename=output_path.name,
        headers={"X-Printability-Report": json.dumps(report)},
    )
