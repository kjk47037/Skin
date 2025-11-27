#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format for faster inference.
Run this script once to convert your .pth model to .onnx format.
"""
import os
import sys
import torch
from app.model import load_model_once, _MODEL, _convert_to_onnx

def main():
    model_path = os.environ.get("MODEL_PATH", "models/efficientnet_full_model.pth")
    onnx_path = model_path.replace(".pth", ".onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    if os.path.exists(onnx_path):
        response = input(f"ONNX model already exists at {onnx_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    print(f"Loading PyTorch model from {model_path}...")
    load_model_once()
    
    if _MODEL is None:
        print("Error: Failed to load model")
        sys.exit(1)
    
    print(f"Converting to ONNX format...")
    try:
        _convert_to_onnx(_MODEL, model_path, onnx_path)
        print(f"✓ Successfully converted model to {onnx_path}")
        print(f"  File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        print("\nThe API will automatically use ONNX Runtime on next startup for faster inference.")
    except Exception as e:
        print(f"Error: Failed to convert model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

