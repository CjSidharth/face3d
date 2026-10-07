# face3d — Photo → 3D-Printable STL Bust

## Goal

Turn one frontal face photo into a 3D-printable STL bust. v1 ("ugly but printable") is
deliberately narrow: get one valid, watertight STL out the other end of a FastAPI
endpoint. Likeness quality and any UI are explicit non-goals for this milestone.

## Hardware constraint

MacBook Air M5, 24 GB unified memory, macOS — **no CUDA anywhere**. Every dependency
choice below was filtered through: does this run on PyTorch's MPS backend or CPU
without a CUDA-only extension?

## Model evaluation & decision

Evaluated for the reconstruction step:

| Model | Topology | MPS-friendly | Hard deps | License |
|---|---|---|---|---|
| DECA / EMOCA | FLAME (clean, full head+neck) | ✗ needs PyTorch3D | PyTorch3D | ✗ Non-commercial (FLAME) |
| MICA | FLAME (metric identity) | ✗ needs PyTorch3D for render | PyTorch3D | ✗ Non-commercial (FLAME) |
| Pixel3DMM | FLAME | ✗✗ PyTorch3D + nvdiffrast, optimization-based | PyTorch3D + nvdiffrast | ✗ Non-commercial (FLAME) |
| FaceLift (ICCV'25) | Gaussian splats → mesh, 360° head | ~ diffusion + GS-LRM, CUDA rasterizer | diff-gaussian-rasterization | Code Apache-2.0 / **weights Adobe non-commercial** ✗ |
| **TripoSR** | Triplane NeRF → marching-cubes solid | **✓ runs on MPS** | torchmcubes (replaced, see below) | **✓ MIT — commercial OK** |

**Decision: TripoSR.** It's the only candidate satisfying every hard constraint —
MPS-native, no PyTorch3D/nvdiffrast, MIT-licensed end to end (code + weights), and its
marching-cubes output is a single closed-ish solid rather than a FLAME mesh's separate
eyeball spheres / open mouth cavity / open neck, which makes the printability layer
simpler to build.

**Tradeoff, stated plainly:** TripoSR is a general single-image-to-3D model, not a face
specialist. Likeness/identity will be noticeably weaker than a FLAME-based
reconstruction, and the back of the head can flatten or lose detail since it's inferring
unseen geometry. That's an accepted cost for "ugly but printable" v1 — the
reconstruction step is isolated behind an interface (see Architecture) so it can be
swapped for a FLAME model or a remote-GPU FaceLift backend later without touching
preprocessing, printability, or the API.

### Licensing flags (commercial use)

- **TripoSR** — MIT (code + weights). Commercial use OK.
- **DECA / EMOCA / MICA / Pixel3DMM** — all built on **FLAME**, which is
  **non-commercial only**. Do not ship these in anything commercial without contacting
  `ps-license@tue.mpg.de`.
- **FaceLift** — code is Apache-2.0, but the released **model weights are under the
  Adobe Research License** (non-commercial). The code alone isn't useful without weights.
- **rembg** (background removal) — verify the specific ONNX model in use (default
  U²-Net) is acceptable for your commercial context before shipping; swap models if not.

## Apple Silicon specifics

- **`torchmcubes` is not used.** TripoSR's official code depends on it for marching
  cubes, and it has no prebuilt wheel and doesn't target Metal/MPS — on macOS it either
  fails to build or silently compiles a CPU-only fallback that's 60x+ slower with no
  obvious error. We vendor TripoSR's `tsr` inference package
  (`pipeline/reconstruct/vendor/tsr/`, MIT-licensed) with `models/isosurface.py` patched
  to use `scikit-image`'s marching cubes instead — no compilation, same result. See
  `pipeline/reconstruct/vendor/tsr/README_VENDOR.md` for the exact diff and rationale.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** is set at process start (`app/config.py`) so any
  op TripoSR's transformer backbone needs that isn't yet implemented on MPS silently
  falls back to CPU instead of crashing.
- Device selection (`resolve_device()` in `app/config.py`) picks `mps` when available,
  else `cpu`. There is no `cuda` branch in this codebase.
- Expect **~30–90 seconds per reconstruction** on M-series (TripoSR benchmarks
  sub-second on an A100; there's no free lunch without CUDA). This is expected, not a bug.
- Use a **Python 3.10+ venv** — the system Python here is 3.9, and the MPS-enabled
  PyTorch wheels target 3.10+. Built/tested on 3.10.20 (via pyenv, since no 3.11+
  was installed on this machine).

## Architecture

```
face3d/
  app/
    main.py                    # FastAPI app + POST /generate
    config.py                  # settings, device selection, MPS fallback env var
  pipeline/
    preprocess.py               # face detect -> crop to head+shoulders -> bg removal
    reconstruct/
      base.py                   # ReconstructionBackend protocol
      triposr_backend.py        # TripoSR on MPS/CPU, image -> trimesh.Trimesh
      vendor/tsr/                # vendored+patched TripoSR inference code (MIT)
    printability.py             # watertight/manifold repair, close base, pedestal, mm scale
    export.py                   # write STL, verify watertight, return a report
  models/                       # downloaded TripoSR weights (gitignored)
  outputs/                      # generated STLs (gitignored)
  tests/
    test_printability.py
    test_export.py
```

### Reconstruction seam

```python
class ReconstructionBackend(Protocol):
    def reconstruct(self, image: PIL.Image.Image) -> trimesh.Trimesh: ...
```

TripoSR is one implementation. Swapping in a FLAME backend, or a backend that calls out
to a remote GPU (for a CUDA-only model), means writing a new class against this
protocol — nothing downstream changes.

## Pipeline (end to end)

1. **Input** — `POST /generate`, multipart upload (JPEG/PNG) + optional `height_mm`
   form field (default 100).
2. **Preprocess** (`pipeline/preprocess.py`)
   - Face detect (MediaPipe FaceDetection, CPU) → crop to head + shoulders, padded for
     a bust, not just a tight face crop.
   - Background removal (`rembg`, ONNX/CPU) → what TripoSR expects as input.
3. **Reconstruct** (`pipeline/reconstruct/triposr_backend.py`)
   - Load `stabilityai/TripoSR` via the vendored `tsr` package, run on `device="mps"`.
   - `model.extract_mesh(...)` → skimage-based marching cubes → `trimesh.Trimesh`
     (geometry only; no texture baking, STL doesn't carry color anyway).
4. **Printability** (`pipeline/printability.py`) — the actual value-add:
   - Repair with `trimesh` + `pymeshlab`: merge close vertices, drop degenerate/duplicate
     faces, fix normals/winding, fill holes → watertight + manifold.
   - Orient upright, cut a base plane, close the neck/back into a flat bottom.
   - Add a flat pedestal base for print stability.
   - Scale to the requested real-world height in mm (default 100 mm).
5. **Export & verify** (`pipeline/export.py`)
   - Write binary STL via `trimesh`.
   - Verify: `is_watertight`, `is_winding_consistent`, `euler_number`, positive
     `volume` — re-load the written STL and re-check (catches export bugs, not just
     in-memory mesh bugs).
6. **Response** — the STL file plus a JSON verification report.

## Milestones

1. Scaffold repo, `requirements.txt`, `config.py`, docs. **(done)**
2. Preprocess module (detect/crop/bg-removal).
3. TripoSR backend on MPS with skimage marching cubes.
4. Printability module (watertight, neck/back close, pedestal, mm scale).
5. Export + watertight verification + unit tests.
6. FastAPI `/generate` endpoint wiring it together.

## Verification

- **Unit:** `tests/test_printability.py` — a repaired mesh is watertight, manifold,
  positive volume. `tests/test_export.py` — STL round-trips through disk and re-verifies.
- **Manual:** `curl -F file=@face.jpg -F height_mm=100 http://localhost:8000/generate`
  → downloads a `.stl` and a report with `is_watertight: true`.
- **Print-sanity:** open the STL in a slicer — one closed solid, correct height, flat
  base sits on the print bed, no non-manifold edges/holes flagged.

## Known v1 limitations (by design)

- Likeness is rough — TripoSR is a general reconstruction model, not a face specialist.
- Back-of-head detail is inferred, not observed (single frontal photo).
- No UI; API only, synchronous (a few tens of seconds per request).
- No auth/rate-limiting on the endpoint — local/dev use only as shipped.
