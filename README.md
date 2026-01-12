# YOLO12 VisDrone Project

This repository contains code and trained models for drone/object detection using YOLO12 on the VisDrone dataset.

## Project Structure

- `dji_async_pro_1280.py`, `dji_async_pro_960.py`: Main inference scripts for DJI drone applications
- `dji_turbo_multiprocessing.py`: Multiprocessing implementation for optimized performance
- `train_visdrone_yolo12.py`: Training script for YOLO12 on VisDrone dataset
- `test.yaml`: Configuration file for testing
- `yolo11n.pt`, `yolo12n.pt`: Pre-trained model weights
- `DJI_VisDrone/`: Contains training results, weights, and evaluation metrics

## Key Features

- Optimized for drone-based object detection
- Includes multiple trained models with different resolutions (1280, 960)
- Evaluation metrics and visualization tools
- Multiprocessing support for real-time inference

## Usage

Run the inference scripts to perform detection on video streams or image sequences:

```bash
python dji_async_pro_1280.py
```

or

```bash
python dji_async_pro_960.py
```

## Contents

The `DJI_VisDrone/` directory contains:

- Trained model weights
- Training logs and metrics
- Evaluation results (confusion matrices, precision/recall curves)
- Sample output images