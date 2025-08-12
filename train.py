import argparse
from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on cow muzzle dataset")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial weights to finetune")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'mps', or CUDA index like '0'")
    parser.add_argument("--name", default="yolo11-cow", help="Run name (folder under runs/)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    args = parser.parse_args()

    # Resolve device: prefer CUDA, then MPS, else CPU if 'auto'
    dev = args.device
    if isinstance(dev, str) and dev.lower() in {"auto", "cuda", "gpu"}:
        if torch.cuda.is_available():
            dev = "0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    # Initialize from pretrained weights
    model = YOLO(args.weights)

    # Train on the dataset defined in data/data.yaml
    model.train(
        data="data/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name=args.name,
        pretrained=True,
        device=dev,
        patience=args.patience,
    )

    # Validate the best checkpoint
    model.val(data="data/data.yaml", device=dev)


if __name__ == "__main__":
    main()
