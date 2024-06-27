# POC Object & Speed Detection

Repo which serves as a POC for object and speed detection using a fixed Region of Interest which has been manually measured.

<p align="center">
  <img src="./data/demo.gif" alt="Demo" height=400px/>
</p>

## 🪒 Setup

Using conda or virtualenv install the packages.

```bash
conda create --name <YOUR_ENV_NAME> python=3.10
conda activate <YOUR_ENV_NAME>
pip install -r requirements.txt
```

## 🐍 Usage

### Speed Estimation

- Create a Region of Interest (ROI) and find its coordinates and dimensions (width,height) in meters
- instantiate the src.speed_estimator.SpeedEstimator() with the video filepath, ROI coordinates (a polygon) and its dimensions.
- Run the SpeedEstimator().estimate_speed() method, see an example in `src/speed_estimator.py`

### Detection

Meant to be used as a POC to integrate classifiers on top of an object detector output using cameras via RTSP or video inputs

- Implements a Yolo Based Detector
- OpenCV detectors: haarcascade_profileface, hog and a classifier on top of its detections which uses the ONNX protocol. Examples can be found in src/classifiers.py (can be any exported classifier in ONNX format)

## 🤿 Contributing to this repo

- This repo uses pre-commit hooks for code formatting and structure. Before adding|commiting changes, make sure you've installed the pre-commit hook running `pre-commit install` in a terminal. After that changes must be submitted as usual (`git add <FILE_CHANGED> -> git commit -m "" -> git push `)

- For dependencies, [pip-tools](https://github.com/jazzband/pip-tools) is used. Add the latest dependency to the requirements.txt and run  `pip-compile requirements.txt -o requirements.txt` to make sure the requirements file is updated so we can re-install with no package version issues.
