YOLO11 Cow Muzzle Detection

This repo trains and tests a YOLO11 model to detect cow muzzles locally. Face can be added later once annotations are ready.

Setup
- Create and activate a virtualenv:
  - `python -m venv .venv && source .venv/bin/activate`
- Upgrade basics and install dependencies:
  - `python -m pip install --upgrade pip wheel setuptools`
  - `pip install ultralytics`
- Optional (only for later TFLite export):
  - `pip install tensorflow`

Dataset
- The dataset is already in `data/` with the expected YOLO structure:
  - `data/train/images`, `data/train/labels`
  - `data/valid/images`, `data/valid/labels`
  - `data/test/images`, `data/test/labels`
- Config file: `data/data.yaml` (currently 1 class: `Muzzle`).

Scripts
- `train.py`: Fine-tunes YOLO11 on the muzzle dataset.
  - Args: `--weights`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--name`, `--patience`.
  - Device handling: `--device auto` resolves to CUDA (`'0'`) if available, else Apple `mps`, else CPU.
  - Defaults: `--weights yolo11n.pt`, `--epochs 100`, `--imgsz 640`, `--batch -1`, `--device auto`, `--name yolo11-cow`, `--patience 25`.
- `validate.py`: Validates a trained checkpoint on the dataset and reports metrics.
  - Args: `--weights`, `--data`, `--imgsz`, `--batch`, `--device`, `--name`.
  - Defaults: `--weights runs/yolo11-cow/weights/best.pt`, `--data data/data.yaml`, `--imgsz 640`, `--batch -1`, `--device auto`, `--name val`.
- `predict.py`: Runs local inference on images/videos/dirs/URLs or webcam, saves annotated outputs.
  - Args: `--weights`, `--source`, `--imgsz`, `--conf`, `--device`.
  - Defaults: `--weights runs/yolo11-cow/weights/best.pt`, `--source data/valid/images`, `--imgsz 640`, `--conf 0.25`, `--device auto`.
- `main.py` (optional later): Exports a trained model to TFLite for mobile/edge deployment.

Train
- Quick sanity run:
  - `python train.py --epochs 5 --imgsz 640 --device auto`
- Full run example:
  - `python train.py --epochs 50 --imgsz 640 --device auto --weights yolo11n.pt --name yolo11-cow`
  - Note: `--device auto` resolves to CUDA if available, else Apple `mps`, else CPU.
- Outputs:
  - Runs under `runs/yolo11-cow/`
  - Best weights at `runs/yolo11-cow/weights/best.pt`

Validate
- Training script runs validation after training, but you can run it explicitly:
  - `python validate.py --weights runs/yolo11-cow/weights/best.pt --data data/data.yaml --imgsz 640 --device auto --name val --split val`
  - Results and PR curves are written under `runs/val/`.

Test Set (Hold‑out)
- Use the separate test split only for final, unbiased evaluation:
  - `python validate.py --weights runs/yolo11-cow/weights/best.pt --data data/data.yaml --imgsz 640 --device auto --name test --split test`
- You can also run predictions on test images for qualitative checks:
  - `python predict.py --weights runs/yolo11-cow/weights/best.pt --source data/test/images --device auto`
- Tip: Avoid tuning hyperparameters based on test results to prevent overfitting to the test set.

Predict (Local Testing)
- On validation images directory:
  - `python predict.py --weights runs/yolo11-cow/weights/best.pt --source data/valid/images --device auto`
- On a single image:
  - `python predict.py --weights runs/yolo11-cow/weights/best.pt --source path/to/image.jpg`
- On a video file:
  - `python predict.py --weights runs/yolo11-cow/weights/best.pt --source path/to/video.mp4 --device auto`
  - The annotated video is saved under `runs/predict/` with the same base filename.
- Webcam (if available):
  - `python predict.py --weights runs/yolo11-cow/weights/best.pt --source 0`
- Useful flags:
  - `--conf 0.25` set confidence threshold (e.g., 0.20–0.40)
  - `--imgsz 640` inference image size
  - `--device auto|cpu|0` select CPU or GPU
- Outputs are saved under `runs/predict/`.

Notes
- Current dataset has only the `Muzzle` class. To add `Face` later, update `data/data.yaml` to `nc: 2` and `names: ['Muzzle', 'Face']`, ensure labels use class IDs 0/1, and retrain.
- If pip install fails due to extra packages in `requirements.txt`, you can install only what’s needed for local training: `pip install ultralytics`.
- For best small-object performance, consider larger `--imgsz` (e.g., 768 or 960) if GPU memory allows.

Export to TFLite (Later)
- Once local results are satisfactory, export with Ultralytics:
  - `python -c "from ultralytics import YOLO; YOLO('runs/yolo11-cow/weights/best.pt').export(format='tflite')"`
- Then test the `.tflite` model similarly to your `main.py` or via a small comparison script.
