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
- instantiate the src.speed_estimator.SpeedEstimator() with the ROI coordinates (a polygon) and its dimensions.
- provide a video and run the SpeedEstimator().estimate_speed

### Detection

- Meant to be used as a POC to integrate classifiers on top of an object detector output using cameras via RSTP or video inputs


## 🤿 Contributing to this repo

- This repo uses pre-commit hooks for code formatting and structure. Before adding|commiting changes, make sure you've installed the pre-commit hook running `pre-commit install` in a terminal. After that changes must be submitted as usual (`git add <FILE_CHANGED> -> git commit -m "" -> git push `)

- For dependencies, [pip-tools](https://github.com/jazzband/pip-tools) is used. Add the latest dependency to the requirements.txt and run  `pip-compile requirements.txt -o requirements.txt` to make sure the requirements file is updated so we can re-install with no package version issues.
