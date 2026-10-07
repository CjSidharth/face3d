# CLAUDE.md

Guide for working in this repo. See [PLAN.md](PLAN.md) for the full design rationale
(model evaluation, licensing, architecture). This file is the quick-reference for
day-to-day work.

## What this is

A pipeline that turns one frontal face photo into a 3D-printable STL bust:
`preprocess -> reconstruct (TripoSR) -> printability repair -> export STL`, wrapped in
a FastAPI endpoint. Runs entirely on Apple Silicon (MPS/CPU) — **no CUDA, no
PyTorch3D, no torchmcubes** anywhere in this codebase. If you're about to add a
dependency that pulls one of those in, stop and check PLAN.md's model table first.

## Hard constraints (do not violate without discussing with the user)

1. **No CUDA-only code paths.** `app/config.py:resolve_device()` only ever returns
   `"mps"` or `"cpu"`. Don't add a `cuda` branch.
2. **No PyTorch3D / nvdiffrast / torchmcubes.** These are why DECA/EMOCA/MICA/
   Pixel3DMM were ruled out and why TripoSR's marching cubes was patched — see
   `pipeline/reconstruct/vendor/tsr/README_VENDOR.md`.
3. **`PYTORCH_ENABLE_MPS_FALLBACK=1`** must be set before `torch` runs any op.
   `app/config.py` sets it as an import side effect — always import `app.config`
   before touching `torch`, don't set the env var ad hoc elsewhere.
4. **Reconstruction models are non-commercial by default.** If you're asked to add or
   swap in a FLAME-based model (DECA/EMOCA/MICA/Pixel3DMM) or FaceLift's weights,
   flag the non-commercial license to the user before wiring it in — see PLAN.md's
   licensing section.

## Setup

```bash
python3.10 -m venv .venv   # system Python here is 3.9; MPS wheels need 3.10+ (this repo was built/tested on 3.10.20 via pyenv)
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads TripoSR weights (`stabilityai/TripoSR`, few hundred MB) from
Hugging Face into the HF cache — expect a delay on first request only.

## Running

```bash
uvicorn app.main:app --reload --port 8000
curl -F "file=@face.jpg" -F "height_mm=100" http://localhost:8000/generate -o bust.stl
```

Reconstruction takes **~30-90s on M-series** (no CUDA) — this is expected, not a hang.

## Tests

```bash
pytest tests/ -v
```

`tests/test_printability.py` and `tests/test_export.py` run on synthetic/sample
meshes, not through the full TripoSR model (too slow for a unit test loop) — they
validate the printability/export contract in isolation.

## Architecture map

- `app/config.py` — settings, device selection, MPS fallback env var. Import this
  before any other `torch`-touching module.
- `app/main.py` — FastAPI app, single `POST /generate` endpoint.
- `pipeline/preprocess.py` — face detect (MediaPipe) → crop to head+shoulders →
  background removal (`rembg`).
- `pipeline/reconstruct/base.py` — `ReconstructionBackend` protocol
  (`image -> trimesh.Trimesh`). **This is the seam** for swapping models later
  (remote-GPU backend, FLAME backend, etc.) — implement against this protocol,
  don't special-case a new model into `printability.py` or `main.py`.
- `pipeline/reconstruct/triposr_backend.py` — the TripoSR implementation of that
  protocol.
- `pipeline/reconstruct/vendor/tsr/` — vendored TripoSR inference code (MIT), with
  `models/isosurface.py` patched to use `scikit-image` instead of `torchmcubes`.
  **Do not `pip install triposr` or re-add `torchmcubes`** — read
  `vendor/tsr/README_VENDOR.md` before touching anything in this directory.
- `pipeline/printability.py` — the actual value-add: watertight/manifold repair,
  closing the neck/back, adding a pedestal, scaling to real mm. Takes/returns a plain
  `trimesh.Trimesh` — has no knowledge of which reconstruction backend produced it.
- `pipeline/export.py` — writes the STL and verifies it (watertight, manifold,
  positive volume), re-checking after the disk round-trip.

## Conventions

- Every stage passes a plain `trimesh.Trimesh` to the next — don't leak
  backend-specific types (e.g. TripoSR's `scene_codes`) past `reconstruct/`.
- Config values (target height, MC resolution, chunk size, pedestal size) live in
  `app/config.py:Settings` — don't hardcode magic numbers in pipeline modules.
- STL is geometry-only; don't add texture/color plumbing to the export path.
