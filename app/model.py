import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

try:
	import requests  # optional; used for remote model download
except Exception:  # pragma: no cover
	requests = None  # type: ignore

try:
	import onnxruntime as ort
	ONNX_AVAILABLE = True
except ImportError:
	ONNX_AVAILABLE = False
	ort = None  # type: ignore


_MODEL: Optional[torch.nn.Module] = None
_ONNX_SESSION: Optional[object] = None  # ONNX Runtime session
_DEVICE: Optional[torch.device] = None
_LABELS: Optional[List[str]] = None
_TRANSFORM: Optional[transforms.Compose] = None
_USE_ONNX: bool = False


_DEFAULT_LABELS = [
	# Fill with your canonical class order from training if different
	"Acne",
	"Actinic_Keratosis",
	"Benign_tumors",
	"Bullous",
	"Candidiasis",
	"DrugEruption",
	"Eczema",
	"Infestations_Bites",
	"Lichen",
	"Lupus",
	"Moles",
	"Psoriasis",
	"Rosacea",
	"Seborrh_Keratoses",
	"SkinCancer",
	"Sun_Sunlight_Damage",
	"Tinea",
	"Unknown_Normal",
	"Vascular_Tumors",
	"Vasculitis",
	"Vitiligo",
	"Warts",
]


def get_labels() -> List[str]:
	global _LABELS
	if _LABELS is not None:
		return _LABELS
	labels_env = os.environ.get("LABELS")  # comma-separated list overrides default
	if labels_env:
		_LABELS = [x.strip() for x in labels_env.split(",") if x.strip()]
	else:
		_LABELS = list(_DEFAULT_LABELS)
	return _LABELS


def get_device_name() -> str:
	global _DEVICE
	if _DEVICE is None:
		_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return str(_DEVICE)


def _ensure_model_file(model_path: str) -> None:
	if os.path.exists(model_path):
		return
	download_url = os.environ.get("MODEL_DOWNLOAD_URL")
	if not download_url:
		raise FileNotFoundError(
			f"Model file not found at '{model_path}'. "
			"Either commit the model, mount it at runtime, or set MODEL_DOWNLOAD_URL."
		)
	if requests is None:
		raise RuntimeError("requests is required to download the model but is not installed.")
	pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
	resp = requests.get(download_url, stream=True, timeout=60)
	resp.raise_for_status()
	with open(model_path, "wb") as f:
		for chunk in resp.iter_content(chunk_size=1024 * 1024):
			if chunk:
				f.write(chunk)
	if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
		raise RuntimeError("Downloaded model file appears to be empty.")


def _build_efficientnet_for(num_classes: int) -> torch.nn.Module:
	# Create a vanilla EfficientNet-B0 head for num_classes.
	model = models.efficientnet_b0(weights=None)
	in_features = model.classifier[1].in_features  # type: ignore[index]
	model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore[index]
	return model


def _convert_to_onnx(model: torch.nn.Module, model_path: str, onnx_path: str) -> None:
	"""Convert PyTorch model to ONNX format for faster inference."""
	model.eval()
	dummy_input = torch.randn(1, 3, 224, 224)
	
	try:
		torch.onnx.export(
			model,
			dummy_input,
			onnx_path,
			export_params=True,
			opset_version=11,
			do_constant_folding=True,
			input_names=["input"],
			output_names=["output"],
			dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
		)
	except Exception as e:
		raise RuntimeError(f"Failed to convert model to ONNX: {e}") from e


def load_model_once() -> None:
	global _MODEL, _ONNX_SESSION, _DEVICE, _TRANSFORM, _USE_ONNX
	if _MODEL is not None or _ONNX_SESSION is not None:
		return

	_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	labels = get_labels()
	num_classes = len(labels)

	model_path = os.environ.get("MODEL_PATH", "models/efficientnet_full_model.pth")
	_ensure_model_file(model_path)

	# Try to use ONNX Runtime for faster inference (if available)
	onnx_path = model_path.replace(".pth", ".onnx")
	use_onnx = ONNX_AVAILABLE and os.environ.get("USE_ONNX", "true").lower() == "true"
	
	if use_onnx and os.path.exists(onnx_path):
		# Load ONNX model directly
		try:
			# Configure ONNX Runtime for optimal CPU performance
			sess_options = ort.SessionOptions()
			sess_options.intra_op_num_threads = 0  # Use all available cores
			sess_options.inter_op_num_threads = 0
			sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
			
			_ONNX_SESSION = ort.InferenceSession(
				onnx_path,
				sess_options,
				providers=["CPUExecutionProvider"]
			)
			_USE_ONNX = True
			print(f"Loaded ONNX model from {onnx_path} (faster inference)")
		except Exception as e:
			print(f"Failed to load ONNX model: {e}, falling back to PyTorch")
			use_onnx = False

	# Load PyTorch model (either for inference or for ONNX conversion)
	if not _USE_ONNX:
		model: Optional[torch.nn.Module] = None
		load_error: Optional[Exception] = None

		# 1) TorchScript
		try:
			model = torch.jit.load(model_path, map_location=_DEVICE)
			# Optimize TorchScript model for inference
			try:
				model = torch.jit.optimize_for_inference(model)
			except Exception:
				pass  # Optimization may not be available in all PyTorch versions
		except Exception as exc:  # pragma: no cover
			load_error = exc

		# 2) Full model object or state dict
		if model is None:
			try:
				# PyTorch 2.6+ defaults to weights_only=True which can fail for older checkpoints.
				# We explicitly set weights_only=False assuming the file is trusted.
				obj = torch.load(model_path, map_location=_DEVICE, weights_only=False)
				if isinstance(obj, dict):
					state_dict = obj.get("model_state_dict") or obj.get("state_dict") or obj
					tmp = _build_efficientnet_for(num_classes)
					tmp.load_state_dict(state_dict, strict=False)
					model = tmp
				else:
					# If author saved torch.save(model)
					model = obj
			except Exception as exc:  # pragma: no cover
				load_error = exc

		if model is None:
			# Provide meaningful error
			raise RuntimeError(f"Failed to load model from '{model_path}': {load_error}")

		model.eval()
		_MODEL = model.to(_DEVICE)
		
		# PyTorch uses all available CPU cores by default, no need to set explicitly
		
		# Try to convert to ONNX for future faster inference
		if use_onnx and not os.path.exists(onnx_path):
			try:
				print(f"Converting model to ONNX format (one-time conversion)...")
				# Ensure model is on CPU for ONNX conversion
				model_cpu = model.cpu() if _DEVICE.type == "cuda" else model
				_convert_to_onnx(model_cpu, model_path, onnx_path)
				print(f"ONNX model saved to {onnx_path}. Restart to use ONNX Runtime.")
			except Exception as e:
				print(f"ONNX conversion failed (non-fatal): {e}")

	_TRANSFORM = transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def _ensure_loaded() -> None:
	if _MODEL is None and _ONNX_SESSION is None:
		load_model_once()


def _preprocess(image: Image.Image) -> torch.Tensor:
	assert _TRANSFORM is not None
	assert _DEVICE is not None
	rgb = image.convert("RGB")
	tensor = _TRANSFORM(rgb).unsqueeze(0)  # add batch dimension
	return tensor.to(_DEVICE)


def predict_image_pil(image: Image.Image, top_k: int = 5) -> Dict:
	"""
	Run inference on a PIL image and return a JSON-serializable result containing:
	- predicted_label, score
	- top_k (list of {label, score})
	- probabilities (dict of label -> score)
	
	Note: "Acne" is excluded from all predictions.
	"""
	_ensure_loaded()
	labels = get_labels()
	
	# Exclude "Acne" from predictions
	EXCLUDED_LABELS = {"Acne"}
	
	# Preprocess image
	rgb = image.convert("RGB")
	tensor = _TRANSFORM(rgb).unsqueeze(0)  # add batch dimension
	
	# Run inference using ONNX Runtime (faster) or PyTorch
	if _USE_ONNX and _ONNX_SESSION is not None:
		# ONNX Runtime inference (much faster)
		input_array = tensor.numpy().astype(np.float32)
		outputs = _ONNX_SESSION.run(None, {"input": input_array})
		logits = outputs[0][0]
		# Softmax
		exp_logits = np.exp(logits - np.max(logits))
		probs = exp_logits / exp_logits.sum()
	else:
		# PyTorch inference
		assert _MODEL is not None
		with torch.no_grad():
			input_tensor = tensor.to(_DEVICE)
			logits = _MODEL(input_tensor)
			probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
	
	# Create a mask to exclude Acne
	valid_indices = [i for i in range(len(labels)) if labels[i] not in EXCLUDED_LABELS]
	
	if not valid_indices:
		raise RuntimeError("No valid labels remaining after exclusions")
	
	# Get top prediction excluding Acne
	valid_probs = np.array([probs[i] for i in valid_indices])
	top_valid_idx = int(np.argmax(valid_probs))
	predicted_idx = valid_indices[top_valid_idx]
	
	# Get top_k excluding Acne
	top_k = int(max(1, min(top_k, len(valid_indices))))
	valid_sorted_indices = sorted(valid_indices, key=lambda i: probs[i], reverse=True)[:top_k]
	
	# Build probabilities dict excluding Acne
	probabilities = {labels[i]: float(probs[i]) for i in range(len(labels)) if labels[i] not in EXCLUDED_LABELS}
	
	result = {
		"predicted_label": labels[predicted_idx],
		"score": float(probs[predicted_idx]),
		"top_k": [{"label": labels[i], "score": float(probs[i])} for i in valid_sorted_indices],
		"probabilities": probabilities,
	}
	return result


