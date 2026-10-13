# BDD100K Object Detection and Tracking

A computer vision project focused on object detection and tracking in road scenes using YOLO and the BDD100K dataset.

## Overview

This project aims to detect road users and vehicles in traffic scenes and track objects across video frames.

## Features

- Object detection using YOLO
- Support for image and video inference
- Multi-object tracking using ByteTrack
- Object counting based on tracking IDs
- Model training and evaluation

## Dataset

This project uses the BDD100K dataset.

The dataset is not included in this repository because of its large size.

## Project Structure

```text
.
├── assets/
├── data/
├── sample/
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── track.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The training, detection, and tracking scripts will be documented here as the project implementation is completed.

## Technologies

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- ByteTrack

## License

See the [LICENSE](LICENSE) file for details.
