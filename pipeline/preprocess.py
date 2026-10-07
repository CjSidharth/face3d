"""Face detect -> crop to head+shoulders -> background removal.

Output is backend-agnostic: an RGBA PIL.Image with the subject cut out onto a
transparent background, cropped to a head-and-shoulders "bust" framing (not just a
tight face box, so there's a neck/shoulders to reconstruct). Any model-specific final
formatting (e.g. TripoSR's fixed-ratio foreground resize + gray composite) belongs in
the reconstruction backend, not here — this module's job is generic image prep.

Note on mediapipe: the legacy `mediapipe.solutions.face_detection` API has been
removed from current mediapipe releases (verified against the installed version —
`mediapipe.solutions` no longer exists). We use the current Tasks API
(`mediapipe.tasks.python.vision.FaceDetector`), which — unlike the old API — needs an
explicit `.tflite` model file. It's fetched once from Google's official MediaPipe
model bucket and cached under `models/` (Apache-2.0 licensed, same as MediaPipe
itself; commercial-safe), the same pattern used for TripoSR weights.
"""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.config import settings  # sets U2NET_HOME as an import side effect; import first

import numpy as np
import PIL.Image
import rembg

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)
_FACE_MODEL_PATH = settings.models_dir / "blaze_face_full_range.tflite"


class NoFaceDetectedError(RuntimeError):
    """Raised when no face can be found in the input image."""


@dataclass
class CropBox:
    left: float
    top: float
    right: float
    bottom: float


_face_detector = None
_rembg_session = None


def _ensure_face_model() -> Path:
    if not _FACE_MODEL_PATH.exists():
        tmp_path = _FACE_MODEL_PATH.with_suffix(".tflite.part")
        urllib.request.urlretrieve(_FACE_MODEL_URL, tmp_path)
        tmp_path.rename(_FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


def _get_face_detector() -> vision.FaceDetector:
    global _face_detector
    if _face_detector is None:
        model_path = _ensure_face_model()
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5,
        )
        _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = rembg.new_session()
    return _rembg_session


def detect_face_box(image: PIL.Image.Image) -> CropBox:
    """Detect the primary face and return its bounding box in pixel coordinates.

    If multiple faces are detected, the largest (by box area) is used, on the
    assumption that a bust photo's subject is the dominant face in frame.
    """
    rgb = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_face_detector().detect(mp_image)
    if not result.detections:
        raise NoFaceDetectedError(
            "No face detected in the uploaded image. Use a clear, front-facing photo "
            "with the face unobstructed and reasonably well lit."
        )

    def area(detection) -> int:
        box = detection.bounding_box
        return box.width * box.height

    best = max(result.detections, key=area)
    box = best.bounding_box
    return CropBox(
        left=box.origin_x,
        top=box.origin_y,
        right=box.origin_x + box.width,
        bottom=box.origin_y + box.height,
    )


def crop_to_bust(
    image: PIL.Image.Image,
    face_box: CropBox,
    *,
    side_pad_ratio: float = 0.9,
    top_pad_ratio: float = 0.6,
    bottom_extend_ratio: float = 2.2,
) -> PIL.Image.Image:
    """Expand a face bounding box into a head+shoulders "bust" crop.

    Padding is expressed relative to the face box's own width/height (not the image's
    absolute resolution) so the crop scales sensibly whether the face fills the frame
    or is a small part of a wider photo. `bottom_extend_ratio` is larger than the
    others because it needs to reach past the chin down to the shoulders/chest to give
    the reconstruction model something to build a neck and base from.
    """
    w, h = image.size
    face_w = face_box.right - face_box.left
    face_h = face_box.bottom - face_box.top

    left = int(max(0, face_box.left - side_pad_ratio * face_w))
    right = int(min(w, face_box.right + side_pad_ratio * face_w))
    top = int(max(0, face_box.top - top_pad_ratio * face_h))
    bottom = int(min(h, face_box.bottom + bottom_extend_ratio * face_h))

    if right <= left or bottom <= top:
        raise ValueError(
            "Computed crop box is degenerate (face too close to an image edge); "
            "try a photo with more headroom and shoulder room around the face."
        )

    return image.crop((left, top, right, bottom))


def remove_background(image: PIL.Image.Image) -> PIL.Image.Image:
    """Cut the subject out onto a transparent background (RGBA)."""
    return rembg.remove(image.convert("RGB"), session=_get_rembg_session())


def preprocess(image: PIL.Image.Image) -> PIL.Image.Image:
    """Full preprocessing: detect face, crop to a bust framing, remove background.

    Returns an RGBA image with the subject on a transparent background, cropped to
    head+shoulders. Raises NoFaceDetectedError if no face is found in `image`.
    """
    face_box = detect_face_box(image)
    cropped = crop_to_bust(image, face_box)
    return remove_background(cropped)
