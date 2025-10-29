import torch
from torchvision import models
from pathlib import Path

# ===== CONFIG =====
model_name = "resnet18"
img_size = 128  # Match the training img_size
out_dir = Path(__file__).resolve().parent / "trained" / "classifier"
model_path = out_dir / "best.pt"

# Ensure output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

# ===== Function to Export =====
def export_models(model, device, img_size: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # TorchScript export
    example = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / "best.torchscript.pt"
    traced.save(str(ts_path))
    print(f"✅ TorchScript model saved at: {ts_path}")

    # ONNX export
    onnx_path = out_dir / "best.onnx"
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
    print(f"✅ ONNX model saved at: {onnx_path}")


# ===== Load Model =====
print(f"🔍 Loading model from: {model_path}")

checkpoint = torch.load(model_path, map_location="cpu")

# Initialize base model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = {k: v for k, v in checkpoint.items() if k.startswith("conv1") or k.startswith("layer") or k.startswith("fc")}
else:
    state_dict = checkpoint

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"ℹ️ Missing keys: {missing}")
print(f"ℹ️ Unexpected keys: {unexpected}")

model.eval()

# ===== Export =====
export_models(model, torch.device("cpu"), img_size, out_dir)

print("🎉 Model exported successfully!")
