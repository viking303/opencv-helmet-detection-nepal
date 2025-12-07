# Helmet Detection for Road Safety in Nepal

This project uses OpenCV and YOLO to detect whether motorbike riders are wearing helmets, targeting road safety improvements in Kathmandu and other Nepali cities.

## Why it matters
- Nepal has high motorbike usage and frequent accidents.
- Helmet compliance is critical; automated detection supports awareness and enforcement.

## Tech stack
- OpenCV for image/video processing
- YOLOv5/YOLOv8 for object detection
- Python (Colab for training), optional Streamlit for demo

## Quick start
1. Open the Colab notebook (`notebooks/helmet_detection_nepal.ipynb`).
2. Run the cells to train or evaluate the model.
3. See `examples/` for demo outputs.

## Repository structure
.
├── notebooks/
│   └── helmet_detection_nepal.ipynb   # Your Colab notebook for training & experiments
├── src/
│   ├── infer_video.py                 # Script to run detection on video/webcam
│   └── infer_images.py                # Script to run detection on images
├── data/                              # (Optional) small sample images or metadata
├── examples/
│   ├── screenshots/                   # Save output images with bounding boxes
│   └── demo_videos/                   # Save short demo videos with detection
├── requirements.txt                   # Python dependencies (OpenCV, torch, ultralytics, etc.)
└── LICENSE                            # MIT license for open-source sharing

## Results
- Add precision/recall, confusion matrix, and demo screenshots.
- Note limitations and next steps (larger dataset, domain adaptation).

## Credits
- YOLO model authors and dataset sources listed in the notebook.
- Built for real-world impact in Nepal.
