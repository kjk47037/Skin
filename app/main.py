from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from PIL import Image
import io
import os

from .model import load_model_once, predict_image_pil, get_labels, get_device_name


app = FastAPI(title="Skin Disease Detection API", version="1.0.0")

# Allow requests from anywhere by default; lock this down if needed via ORIGINS env
origins_env = os.environ.get("CORS_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
	# Warm up: load model into memory once
	load_model_once()


@app.get("/", tags=["meta"])
def root() -> JSONResponse:
	return JSONResponse({"message": "Skin Disease Detection API", "docs": "/docs"})


@app.get("/health", tags=["meta"])
def health() -> JSONResponse:
	# If the model loads without exception during startup, we are healthy
	return JSONResponse(
		{
			"status": "ok",
			"device": get_device_name(),
			"num_labels": len(get_labels()),
			"labels": get_labels(),
		}
	)


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
	if file.content_type is None or not file.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Please upload an image file.")

	try:
		image_bytes = await file.read()
		image = Image.open(io.BytesIO(image_bytes))
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

	try:
		result = predict_image_pil(image, top_k=5)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

	return JSONResponse(result)


