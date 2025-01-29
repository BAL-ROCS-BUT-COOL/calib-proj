# **calib-proj**

A Python package for performing **external calibration** of multi-camera systems via the projection of multi-scale markers (MSMs) from a video projector.

TODO : METTRE LIEN ARXIV.
---


## Table of Contents 📋
- [Installation](#installation) 🔧
- [Documentation](#-documentation) 🗎
- [How To Use](#how-to-use) ❔
- [License](#-license) 📃
- [Acknowledgments](#)


### Dependencies (calib-commons) ⛓️
The package depends on the custom utilities Python package 🧰 [`calib-commons`](https://github.com/tflueckiger/calib-commons). To install it:

   ```bash
   # clone the repos
   git clone https://github.com/tflueckiger/calib-commons.git
   cd calib-commons
   # install the package with pip
   pip install .
   # optional: install in editable mode:
   pip install -e . --config-settings editable_mode=strict
   ```

> ⛓️ Dependencies : if the additional dependencies listed in requirements.txt are not satisfied, they will be automatically installed from PyPi. 


### Installation of calib-proj

 ```bash
   # clone the repos
   git clone https://github.com/tflueckiger/calib-proj.git
   cd calib-proj
   # install the package with pip
   pip install .
   # optional: install in editable mode:
   pip install -e . --config-settings editable_mode=strict
   ```
---



## **Documentation** 🗎

### 📝 **Prequisites**
Ensure you have the following before running the package:

 **Internal Camera Parameters (Intrinsics)** ⚙️📷
>💡 This can be done using using the [`calib-commons`](https://github.com/tflueckiger/calib-commons) toolbox, which includes a ['calibrate-intrinsics'](https://github.com/tflueckiger/calib-commons?tab=readme-ov-file#calibrate-intrinsics) command-line tool for automatic internal calibration of multiple cameras using either videos recordings or images. The tool creates a folder containing camera intrinsics in JSON files matching the required format. See here for documentation on how to generate intrinsics with ['calibrate-intrinsics'](https://github.com/tflueckiger/calib-commons?tab=readme-ov-file#calibrate-intrinsics).



## ❔ **How To Use**

### 1. Projection Sequence Generation

To generate a projection sequence in **.mp4** format, run the script `scripts/run_video_generator.py`: 
```bash
   python scripts/run_video_generator.py
```
By default this will create a video of 40s containing 3200 multi-scale markers (MSMs) with 4 scales at 10 fps (each MSM has an exposition time of 100 ms). 

The video will be saved in `video/video.mp4` and a file containing data about the sequence is saved in `video/seq_info.json`.


### 2. Calibration