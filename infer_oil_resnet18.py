#!/usr/bin/env python3
# infer_oil_resnet18.py

import os
import shutil
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def run_inference(input_dir: str, output_dir: str, model_path: str, threshold: float = 0.5):
    """
    Scans input_dir for images, preprocesses them, runs inference,
    and saves copies to output_dir/oil or output_dir/no_oil.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Model Architecture (ResNet18, 1 logit output)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    # 3. Load Trained Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}") from e

    model.to(device)
    model.eval()

    # 4. Define Preprocessing Transforms (Same as Training/Val)
    inference_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 5. Prepare Directories
    input_path = Path(input_dir)
    result_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    oil_dir = result_path / "oil"
    no_oil_dir = result_path / "no_oil"
    oil_dir.mkdir(parents=True, exist_ok=True)
    no_oil_dir.mkdir(parents=True, exist_ok=True)

    # 6. Run Inference
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    print(f"Scanning images in: {input_path}")

    count = 0
    skipped = 0

    with torch.no_grad():
        for file_p in input_path.iterdir():
            if file_p.is_file() and file_p.suffix.lower() in valid_exts:
                try:
                    img = Image.open(file_p).convert("RGB")
                    input_tensor = inference_transform(img).unsqueeze(0).to(device)  # (1,3,H,W)

                    logits = model(input_tensor).squeeze(1)  # (1,)
                    prob = torch.sigmoid(logits).item()

                    if prob > threshold:
                        dest = oil_dir / file_p.name
                        label = "oil"
                    else:
                        dest = no_oil_dir / file_p.name
                        label = "no_oil"

                    shutil.copy2(file_p, dest)
                    count += 1
                    print(f"{file_p.name} -> {label} (p={prob:.4f})")

                except Exception as e:
                    skipped += 1
                    print(f"Skipping {file_p.name}: {e}")

    print(f"Processed {count} images, skipped {skipped}. Results saved to {result_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ResNet18 oil/no_oil inference and copy images into output/oil and output/no_oil."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", required=True, help="Folder to save results (oil/no_oil subfolders).")
    parser.add_argument("--model_path", required=True, help="Path to trained .pt state_dict file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for oil class (default: 0.5).")
    args = parser.parse_args()

    run_inference(args.input_dir, args.output_dir, args.model_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
