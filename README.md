# Bamboo AI

computer vision project for detecting, tracking, and counting bamboo in video.

This repository is still a work in progress. The current goal is to test a pipeline that combines:

- YOLO-based segmentation
- SAHI sliced detection
- ByteTrack-based tracking
- CLAHE image enhancement


## Current Status

This project is not a polished package yet. It is closer to an experimental repo for model training and inference testing.



## Project Files

- [main.py](/Users/jacky/bamboo_ai/main.py) runs the main inference and counting pipeline on a video
- [segmentation.py](/Users/jacky/bamboo_ai/segmentation.py) extracts segmentation regions used before detection
- [sahi_inference.py](/Users/jacky/bamboo_ai/sahi_inference.py) runs sliced object detection with SAHI
- [bytetrack_inference.py](/Users/jacky/bamboo_ai/bytetrack_inference.py) tracks detections and counts unique IDs
- [clahe_inference.py](/Users/jacky/bamboo_ai/clahe_inference.py) applies CLAHE image enhancement
- [train_detect.py](/Users/jacky/bamboo_ai/train_detect.py) trains and evaluates the detection model


## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running Inference

1. Put your trained weights in the expected folders:

- `training_result/detection_small/weights/best.pt`
- `training_result/segment/best.pt`

2. Put your input video inside the project folder.

3. Run:

```bash
python main.py --source your_video_name.mp4
```

Example:

```bash
python main.py --source water.MP4
```

The `--source` value should be the video filename or relative path from the project root.



## Notes

- Device selection now falls back between `mps`, `cuda`, and `cpu`
- The counting zone is based on the current frame size instead of fixed coordinates


## Limitations

- this is still an in-progress research/project repo
- model weights are expected locally and are not packaged cleanly yet
- some scripts are still tailored to the current dataset layout
- there are not yet automated tests

