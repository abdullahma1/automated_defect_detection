import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")


def load_checkpoint(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu")
    model_name = ckpt.get("model", "resnet18")
    img_size = int(ckpt.get("img_size", 224))
    classes = ckpt.get("classes", ["negative", "positive"])
    model = build_model(model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, img_size, classes


def preprocess(image_path: Path, img_size: int):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser(description="Infer binary classifier")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt produced by training")
    parser.add_argument("--image", type=str, required=True, help="Image file to classify")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    image_path = Path(args.image)

    model, img_size, classes = load_checkpoint(weights_path)
    x = preprocess(image_path, img_size)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

    print(json.dumps({
        "image": str(image_path),
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "probs": {classes[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }, indent=2))


if __name__ == "__main__":
    main()

