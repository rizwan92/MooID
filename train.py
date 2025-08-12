import argparse
from pathlib import Path
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
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--ckpt", default="", help="Path to checkpoint (e.g., runs/.../weights/last.pt)")
    parser.add_argument("--save_period", type=int, default=0, help="Save a checkpoint every N epochs (0 disables)")
    parser.add_argument("--exist_ok", action="store_true", help="Allow existing project/name without incrementing")
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

    # Determine checkpoint if resuming
    ckpt_path = None
    if args.resume:
        if args.ckpt:
            cp = Path(args.ckpt)
            if cp.exists():
                ckpt_path = cp
        else:
            # Try to infer last/best from the run name
            last = Path("runs") / args.name / "weights" / "last.pt"
            best = Path("runs") / args.name / "weights" / "best.pt"
            if last.exists():
                ckpt_path = last
            elif best.exists():
                ckpt_path = best

    # Initialize model
    model = YOLO(str(ckpt_path) if ckpt_path else args.weights)

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
        resume=bool(ckpt_path) or args.resume,
        save_period=max(0, args.save_period),
        exist_ok=args.exist_ok,
    )

    # Validate the best checkpoint
    model.val(data="data/data.yaml", device=dev)


if __name__ == "__main__":
    main()
