import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import platform
import contextlib
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Some determinism flags (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FilesDataset(Dataset):
    def __init__(self, files: List[Path], labels: List[int], img_size: int, is_train: bool):
        self.files = files
        self.labels = labels
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = self.labels[idx]
        return x, y


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")


def find_classes_from_folders(root: Path) -> Tuple[List[str], List[Path]]:
    ds = datasets.ImageFolder(root)
    classes = ds.classes  # sorted
    # Gather files per class index
    files = [Path(p) for p, _ in ds.samples]
    labels = [int(y) for _, y in ds.samples]
    return classes, list(zip(files, labels))


def split_files(files_labels: List[Tuple[Path, int]], seed: int, train_ratio=0.7, val_ratio=0.15):
    files = [fl[0] for fl in files_labels]
    labels = [fl[1] for fl in files_labels]
    # First split train vs temp
    f_train, f_tmp, y_train, y_tmp = train_test_split(
        files, labels, test_size=(1 - train_ratio), random_state=seed, stratify=labels
    )
    # Split temp into val and test
    test_ratio = 1 - train_ratio
    val_size = val_ratio / test_ratio
    f_val, f_test, y_val, y_test = train_test_split(
        f_tmp, y_tmp, test_size=(1 - val_size), random_state=seed, stratify=y_tmp
    )
    return (f_train, y_train), (f_val, y_val), (f_test, y_test)


def compute_class_weights(labels: List[int], num_classes: int):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.max() / np.clip(counts, 1, None)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, device, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # CUDA autocast
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU path (no autocast)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = float("nan")
    return running_loss / len(loader.dataset), acc, prec, rec, f1, auc, (all_labels, all_preds, all_probs)


def save_confusion_matrix(labels, preds, out_path: Path, class_names: List[str]):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def export_models(model, device, img_size: int, out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # TorchScript
    example = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / "best.torchscript.pt"
    traced.save(str(ts_path))

    # ONNX
    onnx_path = out_dir / "best.onnx"
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Train binary classifier (positive/negative)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root with positive/negative subfolders")
    parser.add_argument("--out_dir", type=str, default="automated_defect_detection/trained/classifier", help="Output directory")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (Windows/CPU: use 0)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover classes and files
    class_names, files_labels = find_classes_from_folders(data_root)
    if set(class_names) != {"negative", "positive"}:
        class_names = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Remap labels according to current class order
    files, labels = zip(*files_labels)
    labels = list(labels)

    # Split
    (f_train, y_train), (f_val, y_val), (f_test, y_test) = split_files(list(zip(files, labels)), seed=args.seed)

    # Persist split
    split_json = {
        "classes": class_names,
        "train": [str(p) for p in f_train],
        "val": [str(p) for p in f_val],
        "test": [str(p) for p in f_test],
        "seed": args.seed,
    }
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)

    # Datasets and loaders
    ds_train = FilesDataset(f_train, y_train, args.img_size, is_train=True)
    ds_val = FilesDataset(f_val, y_val, args.img_size, is_train=False)
    ds_test = FilesDataset(f_test, y_test, args.img_size, is_train=False)

    # Safer defaults for Windows/CPU
    if args.workers is None:
        if platform.system().lower().startswith("win") or device.type != "cuda":
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)
    else:
        num_workers = max(0, int(args.workers))
    pin_mem = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # Model
    model = build_model(args.model, num_classes=2).to(device)

    # Loss with class weights if imbalance
    class_weights = compute_class_weights(y_train, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    # Use AMP scaler only on CUDA; remove deprecation warnings
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    best_loss = math.inf
    best_f1 = -1.0
    best_state = None
    history = []
    patience = args.patience
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dl_train, device, criterion, optimizer, scaler)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, _ = eval_epoch(model, dl_val, device, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_prec": val_prec,
            "val_rec": val_rec,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        })

        # Early stopping with tie-breaker on F1
        improved = (val_loss < best_loss - 1e-6) or (abs(val_loss - best_loss) <= 1e-6 and val_f1 > best_f1)
        if improved:
            best_loss = val_loss
            best_f1 = val_f1
            best_state = {"model": model.state_dict(), "epoch": epoch}
            patience_left = patience
        else:
            patience_left -= 1

        print(f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} f1 {val_f1:.3f} auc {val_auc:.3f}")
        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    # Save logs
    import csv
    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Save artifacts
    label_map = {0: class_names[0], 1: class_names[1]}
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    torch.save({
        "state_dict": model.state_dict(),
        "model": args.model,
        "img_size": args.img_size,
        "classes": class_names,
        "best_val_loss": best_loss,
        "best_val_f1": best_f1,
    }, out_dir / "best.pt")

    # Evaluate on test split
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, (yl, yp, yprobs) = eval_epoch(model, dl_test, device, criterion)
    report = classification_report(yl, yp, target_names=class_names, output_dict=True)
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "report": report,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    save_confusion_matrix(yl, yp, out_dir / "confusion_matrix.png", class_names)

    # Export models (TorchScript + ONNX)
    export_models(model, device, args.img_size, out_dir, args.model)

    # Try quick ONNX load check via OpenCV if available
    try:
        import cv2  # noqa: F401
        _ = cv2.dnn.readNetFromONNX(str(out_dir / "best.onnx"))
        print("Verified: OpenCV loaded ONNX successfully.")
    except Exception as e:
        print(f"Warning: Could not verify ONNX with OpenCV: {e}")

    # Save a short README
    readme = f"""
Binary classifier training artifacts

Classes: {class_names}
Model: {args.model}
Image size: {args.img_size}

Key files:
- best.pt (PyTorch state dict)
- best.onnx (ONNX for OpenCV)
- best.torchscript.pt (TorchScript)
- label_map.json
- metrics.json
- confusion_matrix.png
- train_log.csv
- splits.json

Example inference:
  python scripts/infer_classifier.py --weights {out_dir}/best.pt --image PATH/TO/IMAGE.jpg
"""
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
