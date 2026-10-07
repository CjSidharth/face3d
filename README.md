# face3d

Turn one frontal face photo into a 3D-printable STL bust — runs entirely on Apple
Silicon (MPS/CPU), no CUDA required.

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # 3.10+ needed for MPS wheels; system Python here is 3.9
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

curl -F "file=@face.jpg" -F "height_mm=100" http://localhost:8000/generate -o bust.stl
```

See [PLAN.md](PLAN.md) for the model evaluation/rationale and [CLAUDE.md](CLAUDE.md)
for the architecture map and repo conventions.

**v1 scope:** one "ugly but printable" bust — geometry over likeness. Reconstruction
uses [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (MIT license, MPS-capable).
Likeness quality and a UI are planned for later milestones.
