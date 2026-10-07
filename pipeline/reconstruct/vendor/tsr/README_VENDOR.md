# Vendored: TripoSR `tsr` package

Source: https://github.com/VAST-AI-Research/TripoSR (commit at time of vendoring:
`main` branch, pulled 2026-08-08). License: MIT (`LICENSE_TRIPOSR` in this directory).
Copyright (c) 2024 Tripo AI & Stability AI.

## Why vendored instead of `pip install`

TripoSR is not published on PyPI; upstream instructions are "clone the repo and run
from it." Vendoring the `tsr` inference package gives us a pinned, importable copy
without needing the rest of the repo (CLI script, Gradio demo, texture baking).

## What changed vs. upstream

- **`models/isosurface.py`** — replaced the `torchmcubes` marching-cubes call with
  `skimage.measure.marching_cubes`. `torchmcubes` has no prebuilt wheel and its CUDA
  extension doesn't target Apple Silicon/MPS; on macOS it either fails to build or
  silently falls back to a very slow, easy-to-miss CPU path. See the comment block at
  the top of that file for the coordinate-axis reasoning behind the change.
- **`bake_texture.py`** — removed (unused). It pulls in `xatlas` and `moderngl`
  (needs an OpenGL context) purely for baking a texture atlas; this project exports
  raw geometry to STL, which is textureless, so it was dead weight.

Everything else (the transformer backbone, triplane tokenizer, NeRF renderer, image
preprocessor) is unmodified upstream code — no CUDA-only ops, no `xformers`/`flash-attn`
hard dependency (it uses `torch.nn.functional.scaled_dot_product_attention`, which
runs on MPS).

## How it's imported

`pipeline/reconstruct/triposr_backend.py` adds this `vendor/` directory to `sys.path`
before doing `from tsr.system import TSR`, since `tsr`'s own code does absolute
`importlib.import_module("tsr....")` lookups (see `tsr/utils.py: find_class`) and
therefore needs `tsr` importable as a top-level package name.

## Re-vendoring / updating

If upstream TripoSR changes, re-run the same process: clone upstream, copy `tsr/`
over this directory (keep `LICENSE_TRIPOSR` and this file), delete `bake_texture.py`,
and re-apply the `isosurface.py` patch.
