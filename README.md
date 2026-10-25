# Football Field Detection with YOLOv5

An experimental computer-vision application that uses a custom YOLOv5 model to locate the playing-field region in video footage and visualize a masked region of interest using OpenCV.

## How it works

1. Loads an exported YOLOv5 detection checkpoint from `model/lapangan.pt`.
2. Reads frames from a local video in `source/`.
3. Filters detections by a class index and accumulates recent detection boxes.
4. Applies rectangular masks to create a visual result.

**Method note:** The current code creates a mask from detection **bounding boxes**. It does not implement pixel-accurate semantic segmentation. It also includes Windows-specific `pathlib` compatibility code.

## Requirements

Python, PyTorch, NumPy, OpenCV, the YOLOv5 source code, and a graphical environment for `cv2.imshow`.

## Local usage

```bash
git clone https://github.com/rafli5131/deteksi-lapangan-yolov5.git
cd deteksi-lapangan-yolov5
python -m venv .venv
# Activate the virtual environment for your OS
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5.git
python main.py
```

The script expects `source/source-video-2.mp4` and `model/lapangan.pt`, both referenced relative to the repository root. Review the model's class mapping and local platform compatibility before running.

Press **q** while the OpenCV window has focus to close the preview.

## Repository layout

- `main.py` — frame processing and field visualization.
- `model/` — custom weights.
- `source/` — sample video material.
- `datasheet/` — supporting project materials.

## Limitations

This is an educational prototype; object-class filtering, temporal smoothing, model provenance, frame processing speed, and detection accuracy need independent evaluation before the approach is used in a robotics pipeline.
