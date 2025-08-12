import argparse
from ultralytics import YOLO


def resolve_device(dev_str: str):
    dev = dev_str
    if isinstance(dev, str) and dev.lower() in {"auto", "cuda", "gpu"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"
    return dev


def main():
    parser = argparse.ArgumentParser(description="Validate a YOLO11 model on the dataset")
    parser.add_argument(
        "--weights",
        default="runs/yolo11-cow/weights/best.pt",
        help="Path to .pt checkpoint to validate",
    )
    parser.add_argument(
        "--data", default="data/data.yaml", help="Path to data.yaml"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 auto)")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'mps', or CUDA index like '0'",
    )
    parser.add_argument(
        "--name", default="val", help="Run name for results under 'runs/'"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate (val or test)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        project="runs",
        name=args.name,
    )

    # Print a minimal summary
    try:
        metrics = results.results_dict
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except Exception:
        print("Validation completed. Check the 'runs' directory for results.")


if __name__ == "__main__":
    main()
