# Overview

This is the python implementation of OpenVINO models.

The OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that emulate human vision. 
Based on Convolutional Neural Networks (CNNs), the toolkit extends CV workloads across Intel® hardware, maximizing performance.

[Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit)

# Supported tasks

We support these tasks.

|task                    |model                                     |
|------------------------|------------------------------------|
|detect_face             |face-detection-adas-0001            |
|emotion_recognition     |emotions-recognition-retail-0003    |
|estimate_headpose       |head-pose-estimation-adas-0001      |
|detect_body             |person-detection-retail-0013        |
|estimate_humanpose      |human-pose-estimation-0001          |
|detect_segmentation     |instance-segmentation-seurity-0050  |

Computed results by models are as below.

![supported_models](https://user-images.githubusercontent.com/34574033/63226303-36bc7b80-c213-11e9-8881-74241128e1d3.png)

# Installation

Click [HERE](https://github.com/hampen2929/pyvino/blob/master/INSTALL.md) for installation instructions.

# Notebook samples
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/tree/master/notebook).

# Update
v0.0.1 (2019/08/24)
- Mac, Windows and Ubuntu are supported/


# Directory structure

```

└── pyvino
    ├── model
    |   └── model.py
    ├── detector
    |   └── detector.py
    ├── segmentor
    |   ├── segmentor.py
    |   └── visualizer.py
    ├── util
    |   ├── config.py
    |   ├── testor.py
    |   └── image.py
    └── tests

```

# License
This project is released under the [Apache 2.0 license](https://github.com/hampen2929/pyvino/blob/master/LICENSE).

# Reference
[OpenCV python demos](https://github.com/opencv/open_model_zoo/tree/master/demos/python_demos)

[OpenCV dnn](https://github.com/opencv/opencv/tree/master/samples/dnn)

[MODEL_ZOO](https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/)
