﻿# 3d_Human_BVH_model_from_video
# Overview
This project implements an automated pipeline for processing 3D motion capture data. It aims to convert raw motion data captured from videos into a usable BVH (Biovision Hierarchy) format while ensuring the quality and realism of the animations. By utilizing advanced smoothing techniques and the SMPL (Skinned Multi-Person Linear) model, this pipeline addresses common issues related to noise and inaccuracies in motion data.

## Features
Data Acquisition: Capture 3D joint data from video recordings.
Preprocessing:
Utilizes the SMPL model for standardized motion representation.
Data cleaning and normalization to enhance consistency.
Noise Reduction:
Implements Kalman filtering, Gaussian smoothing, and low-pass filtering to reduce noise while preserving critical motion details.
BVH Conversion:
Exports processed motion data to BVH format for easy integration with animation software like Blender.
# Getting Started
## Prerequisites
Python 3.11.X and Above
Required libraries:
NumPy
Pandas
SciPy
SMPL model library
Blender (for BVH integration)
## Installation
Download the model From [Download the model] https://drive.google.com/drive/folders/1LQKXwMeMKhF3TirB3--e7xlIBQJ1lKOI?usp=sharing
Clone the repository:

```bash
Copy code

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
### Install required packages:

bash
```Copy code
pip install -r requirements.txt
Usage
Place your video files in the designated input folder.
```
### Run the main script to process the motion data:

```bash
Copy code
python main.py
```
The processed BVH file will be generated in the output folder.

### Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, please create a pull request or open an issue.


### Acknowledgments
Special thanks to the creators of the SMPL model and  the research papers that guided the development of this project.
