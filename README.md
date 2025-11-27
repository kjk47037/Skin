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
   - Railway auto-detects Python and uses `Procfile`
   - Set environment variables (Settings → Variables) as needed:
     - `MODEL_PATH` (default: `models/efficientnet_full_model.pth`)
     - `MODEL_DOWNLOAD_URL` (if not committing the model to Git)
     - `LABELS` (optional comma-separated labels in training order)
     - `CORS_ORIGINS` (optional, e.g. `https://your-frontend.com`)

3) If your `.pth` is larger than GitHub's 100MB limit, use one of:
   - Git LFS for the `models/*.pth` file, or
   - Host the file (S3/GDrive/direct HTTP) and set `MODEL_DOWNLOAD_URL`

4) After deploy, open the public URL, then visit `/docs` for the interactive Swagger UI.

### Notes

- The loader tries TorchScript first, then falls back to `torch.load`. If the file is a state dict, it builds an EfficientNet-B0 head and loads with `strict=False`. If your training architecture differs, update `app/model.py` accordingly.
- The default image preprocessing uses 224x224 ImageNet normalization. For best accuracy, match your training-time transforms.


