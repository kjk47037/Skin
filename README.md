## Skin Disease Detection API (FastAPI + Railway)

This repository serves a PyTorch skin disease classifier behind a FastAPI endpoint suitable for deployment on Railway.

### Project layout

```
app/
  main.py      # FastAPI app (health + /predict)
  model.py     # Model loading and inference utilities
models/
  efficientnet_full_model.pth  # your trained model (or download at runtime)
requirements.txt
Procfile
Dockerfile     # Optimized for Railway (CPU-only PyTorch to reduce image size)
.dockerignore  # Excludes unnecessary files from Docker build
```

### Run locally

1) Create a virtualenv and install dependencies:
```
pip install -r requirements.txt
```

2) Ensure your model is available:
- Option A: place it at `models/efficientnet_full_model.pth`
- Option B: set `MODEL_DOWNLOAD_URL` to a direct URL; it will download at startup

3) Optionally, override labels and paths:
```
export LABELS="Acne,Actinic_Keratosis,Benign_tumors,..."  # comma-separated
export MODEL_PATH="models/efficientnet_full_model.pth"
```

4) Start the server:
```
uvicorn app.main:app --reload --port 8000
```

5) Open docs: http://localhost:8000/docs

### API

- GET `/health` → returns status, device, and labels
- POST `/predict` (multipart file `file`) → returns predicted label and probabilities

### Deploy to Railway

1) Push to GitHub:
```
git init
git add .
git commit -m "Initial API"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

2) On Railway:
   - New Project → Deploy from GitHub → choose this repo
   - Railway will detect the `Dockerfile` and use it (CPU-only PyTorch keeps image < 4GB)
   - Set environment variables (Settings → Variables) as needed:
     - `MODEL_PATH` (default: `models/efficientnet_full_model.pth`)
     - `MODEL_DOWNLOAD_URL` (if not committing the model to Git)
     - `LABELS` (optional comma-separated labels in training order)
     - `CORS_ORIGINS` (optional, e.g. `https://your-frontend.com`)

3) If your `.pth` is larger than GitHub's 100MB limit, use one of:
   - Git LFS for the `models/*.pth` file, or
   - Host the file (S3/GDrive/direct HTTP) and set `MODEL_DOWNLOAD_URL`

4) After deploy, open the public URL, then visit `/docs` for the interactive Swagger UI.

### Performance Optimization

**ONNX Runtime for Faster Inference:**
- The API automatically converts your PyTorch model to ONNX format on first startup (if `USE_ONNX=true` is set, which is the default)
- ONNX Runtime provides **2-5x faster inference** compared to PyTorch CPU
- The converted `.onnx` file will be saved next to your `.pth` file
- On subsequent startups, the API will automatically use the ONNX model for faster inference
- To manually convert: `python convert_to_onnx.py`
- To disable ONNX: set environment variable `USE_ONNX=false`

### Notes

- **Image Size**: The Dockerfile uses CPU-only PyTorch (much smaller than CUDA version) to keep the image under Railway's 4GB limit. ONNX Runtime adds minimal size (~50MB) but provides significant speedup.
- **Inference Speed**: ONNX Runtime is 2-5x faster than PyTorch CPU inference while keeping the image size well under 4GB.
- The loader tries ONNX Runtime first (if available), then TorchScript, then falls back to `torch.load`. If the file is a state dict, it builds an EfficientNet-B0 head and loads with `strict=False`. If your training architecture differs, update `app/model.py` accordingly.
- The default image preprocessing uses 224x224 ImageNet normalization. For best accuracy, match your training-time transforms.
- **Acne exclusion**: The API filters out "Acne" from all predictions (it will never appear in results).


