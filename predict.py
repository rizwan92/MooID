import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11 local inference")
    parser.add_argument(
        "--weights",
        default="runs/yolo11-cow/weights/best.pt",
        help="Path to .pt weights",
    )
    parser.add_argument(
        "--source",
        default="data/valid/images",
        help="Image/video path, directory, URL, or webcam index",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--device", default="auto", help="'auto', 'cpu', 'mps', or CUDA index like '0'"
    )
    args = parser.parse_args()

    # Resolve device: prefer CUDA, then MPS, else CPU if 'auto'
    dev = args.device
    if isinstance(dev, str) and dev.lower() in {"auto", "cuda", "gpu"}:
        try:
            import torch
            if torch.cuda.is_available():
                dev = "0"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        except Exception:
            dev = "cpu"

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=dev,
        save=True,
        project="runs",
        name="predict",
    )

    # Report where predictions are saved
    if results and hasattr(results[0], "save_dir"):
        print(f"Predictions saved to: {results[0].save_dir}")
    else:
        print("Prediction finished.")


if __name__ == "__main__":
    main()
